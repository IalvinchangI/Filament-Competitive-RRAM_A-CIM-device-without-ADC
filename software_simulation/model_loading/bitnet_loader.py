import torch
import numpy
from transformers import AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from threading import Thread
import traceback
import gc

from model_loading import BasicModelLoader
from models import BitLinear
from utils import LoggingColor
from utils.environment_variables import HP_TOKEN, TORCH_DEVICE

from torch import nn
from typing import Union, Iterator
from RRAM_Driver import RRAM_DriverInterface


class TernaryBitNetLoader(BasicModelLoader):
    """
    Ternary BitNet (BitNet b1.58) Loader
    ===========================================================================
    This class inherits from BasicModelLoader to handle the loading, configuration, 
    and execution of BitNet models using ternary weights.

    Key Responsibilities:
    1. Tokenization: Manages text-to-token and token-to-text conversions.
    2. Hardware Mapping: Extracts quantized weights from `BitLinear` layers and 
       programs them into the RRAM hardware driver.
    3. Execution Modes: Supports standard inference, blocking generation, and 
       non-blocking stream generation.
    4. Memory Management: Cleans up PyTorch weight tensors once offloaded to hardware 
       to optimize memory usage.
    
    ===========================================================================
    """

    BIT_DEPTH = 8
    MAX_TOKEN = 50
    FIX_OUTPUT_GENERATION = False
    STREAM_GENERATE_TIMEOUT = None
    
    _tokenizer = None
    _DEFAULT_TOKENIZER_ID = "microsoft/bitnet-b1.58-2B-4T"
    
    GENERATE_COMMAND = "generate_command"
    INFERENCE_COMMAND = "inference_command"
    STREAM_GENERATE_COMMAND = "stream_generate_command"

    _logger = LoggingColor.get_logger("TernaryBitNetLoader")

    @classmethod
    def set_MAX_TOKEN(cls, value: int):
        """
        Set the maximum number of tokens to generate.

        Args:
            value (int): The maximum token count.

        Raises:
            ValueError: If the value is less than or equal to 0.
        """
        if value <= 0:
            raise ValueError("MAX_TOKEN should be greater than 0.")
        cls.MAX_TOKEN = value

    @classmethod
    def init_tokenizer(cls, id: str = None) -> bool:
        """
        Initialize and cache the tokenizer.

        Args:
            id (str, optional): The HuggingFace model ID for the tokenizer. 
                                Defaults to the class default ID.

        Returns:
            bool: True if the tokenizer was successfully loaded or already exists, False otherwise.
        """
        if cls._tokenizer is not None:
            return True
            
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls._DEFAULT_TOKENIZER_ID if (id is None) else id, 
                token=HP_TOKEN
            )
            if cls._tokenizer.pad_token is None:
                cls._tokenizer.pad_token = cls._tokenizer.eos_token
            return True
        except Exception as e:
            cls._tokenizer = None
            cls._logger.error(LoggingColor.color_text(f"Failed to load tokenizer: {e}", LoggingColor.ERROR))
            return False

    @classmethod
    def str2token(cls, string: str) -> numpy.ndarray:
        """
        Convert a string into token IDs.

        Args:
            string (str): The input text string.

        Returns:
            numpy.ndarray: An array of encoded token IDs.
        """
        if cls._tokenizer is None:
            if cls.init_tokenizer() == False:
                return numpy.array([])

        encoded = cls._tokenizer(string, return_tensors="np")
        return encoded.input_ids

    @classmethod
    def token2str(cls, tokens: Union[int, list, numpy.ndarray, torch.Tensor]) -> str:
        """
        Convert token IDs back into a decoded string.

        Args:
            tokens: The token IDs to decode (supports int, list, numpy array, or torch tensor).

        Returns:
            str: The decoded text string.
        """
        if cls._tokenizer is None:
            if cls.init_tokenizer() == False:
                return ""

        if isinstance(tokens, torch.Tensor):
            if tokens.numel() == 1:
                tokens = tokens.item()
            else:
                tokens = tokens.tolist()
        elif isinstance(tokens, numpy.ndarray):
            if tokens.size == 1:
                tokens = tokens.item()
            else:
                tokens = tokens.tolist()
        elif isinstance(tokens, (int, numpy.integer)):
            tokens = int(tokens)
            
        decoded = cls._tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded

    @classmethod
    def print_stream(cls, stream_result: "TextStreamTracker") :
        """
        Directly print the output of a text stream to standard output.

        Args:
            stream_result (TextStreamTracker): The stream generator yielding text chunks.
        """
        if stream_result is None:
            return

        try:
            for new_text in stream_result:
                print(new_text, end="", flush=True)
        except Exception as e:
            if hasattr(stream_result, 'send_stop'):
                stream_result.send_stop()
            cls._logger.error(LoggingColor.color_text(f"Stream decoding error: {e}", LoggingColor.ERROR))

    def __init__(self, model_path: str):
        super().__init__()
        try:
            self._logger.info(f"Loading model from {model_path}")
            self.model: nn.Module = torch.load(model_path, map_location=TORCH_DEVICE, weights_only=False)
            self.model.eval()
            self.device = next(self.model.parameters()).device
            self._logger.info(LoggingColor.color_text("Loaded.", LoggingColor.GREEN))
        except Exception as e:
            self._logger.error(LoggingColor.color_text(f"Load failed: {e}", LoggingColor.ERROR))
            self.model = None

        self.driver: RRAM_DriverInterface = None

    def config(self, driver: RRAM_DriverInterface):
        """
        Configure the hardware driver and offload BitLinear weights.

        Calculates scales, quantizes weights, submits them to the RRAM driver, 
        and sets up the global matrix multiplication adapter.

        Args:
            driver (RRAM_DriverInterface): The hardware driver instance.
        """
        super().config(driver)
        
        if self.model is None:
            return

        self._logger.info("Configuring BitNet hardware acceleration...")
        count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                module.calculate_weight_scale()
                w_quant_tensor = module.get_quantization_weights().to(torch.int8)
                w_quant_np = w_quant_tensor.cpu().numpy().T
                
                gid = driver.submit_matrix([w_quant_np])
                module.layer_id = gid
                count += 1
                del module.weight
        
        gc.collect()
        torch.cuda.empty_cache()
        
        self._logger.info(f"Configured {count} BitLinear layers to Hardware.")

        def _matmul_hardware_adapter(x_quant_tensor, layer_id: str):
            x_np = x_quant_tensor.cpu().numpy().astype(numpy.int8)
            y_np = driver.compute_multibit(layer_id, x_np, self.BIT_DEPTH)
            y_tensor = torch.from_numpy(y_np).to(device=x_quant_tensor.device)
            return y_tensor

        BitLinear.set_matmul(_matmul_hardware_adapter)

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> Union[numpy.ndarray, "TextStreamTracker"]:
        """
        Execute the BitNet model.

        Supports standard inference, generation, and streaming generation.

        Args:
            input_data (numpy.ndarray): The tokenized input array.
            input_command (str, optional): The execution mode (e.g., `INFERENCE_COMMAND`, 
                                           `STREAM_GENERATE_COMMAND`). Defaults to None.

        Returns:
            Union[numpy.ndarray, Iterator[str]]: 
                - numpy.ndarray: If running standard INFERENCE or GENERATE.
                - TextStreamTracker: If running STREAM_GENERATE (returns a generator).
        """
        super().run(input_data, input_command)

        if self.model is None:
            self._logger.error(LoggingColor.color_text("Model not loaded.", LoggingColor.ERROR))
            return None
        
        input_tensor = torch.from_numpy(input_data).to(self.device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        attention_mask = torch.ones_like(input_tensor)

        self._logger.info(f"Running {input_command}...")

        output_data = None

        try:
            if input_command == self.INFERENCE_COMMAND or input_command is None:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits
                output_data = logits.cpu().numpy()
            
            elif input_command == self.GENERATE_COMMAND:
                with torch.no_grad():
                    output_tensor = self.model.generate(
                        input_tensor, 
                        attention_mask=attention_mask, 
                        max_new_tokens=self.MAX_TOKEN, 
                        pad_token_id=self.model.config.eos_token_id, 
                        do_sample=(not self.FIX_OUTPUT_GENERATION)
                    )
                output_data = output_tensor.cpu().numpy()

            elif input_command == self.STREAM_GENERATE_COMMAND:
                if self._tokenizer is None:
                    if not self.init_tokenizer():
                         return None
                
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=self.STREAM_GENERATE_TIMEOUT)
                tracker = TextStreamTracker(streamer)
                stopping_criteria = StoppingCriteriaList([self.StopSignalCriteria(tracker)])

                generation_kwargs = dict(
                    inputs=input_tensor, 
                    attention_mask=attention_mask, 
                    streamer=streamer, 
                    max_new_tokens=self.MAX_TOKEN, 
                    pad_token_id=self.model.config.eos_token_id, 
                    do_sample=(not self.FIX_OUTPUT_GENERATION), 
                    stopping_criteria=stopping_criteria
                )

                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                return tracker

            else:
                self._logger.warning(LoggingColor.color_text(f"Unknown command: {input_command}", LoggingColor.WARNING))
                return None

        except Exception as e:
            self._logger.error(LoggingColor.color_text(f"Runtime error: {e}", LoggingColor.ERROR))
            traceback.print_exc()
            return None

        return output_data
        
    def unload(self):
        """
        Unload the model and clear matrices from hardware.

        Records:
            - Clears specific Layer IDs associated with BitLinear modules from the hardware.
        """
        super().unload()
        if self.model is None or self.driver is None:
            return

        self._logger.info("Unloading model resources...")
        count = 0
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                if module.layer_id is not None:
                    if self.driver.clear_matrix(module.layer_id):
                        count += 1
                    module.layer_id = None 

        BitLinear.set_matmul(None) 
        self._logger.info(f"Cleared {count} matrices from hardware.")

    def loader_details(self) -> dict:
        """
        Retrieve specific configuration details.

        Returns:
            dict: Configuration dictionary including bit depth, max tokens, etc.
        """
        return {
            "bit_depth": self.BIT_DEPTH, 
            "fix_output_generation": self.FIX_OUTPUT_GENERATION, 
            "max_token": self.MAX_TOKEN
        }

    class StopSignalCriteria(StoppingCriteria):
        def __init__(self, tracker: "TextStreamTracker"):
            self.tracker = tracker
            
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            return self.tracker.stop_signal


class TextStreamTracker:
    """
    Text Stream Tracker
    ===========================================================================
    A wrapper class for TextIteratorStreamer.

    Key Responsibilities:
    1. Stream Observation: Silently accumulates the full generated text in the 
       background while the frontend reads and prints chunks iteratively.
    2. Lifecycle Control: Provides mechanisms to send stop signals and trigger 
       callbacks upon stream completion.
    
    ===========================================================================
    """
    def __init__(self, streamer):
        self.streamer = streamer
        self._accumulated_text = ""
        self._on_finish_callback = None
        self.stop_signal = False
    
    def send_stop(self):
        """Trigger the stop signal to halt generation."""
        self.stop_signal = True
    
    def register_on_finish_callback(self, callback):
        """
        Register a callback function to be executed when the stream ends.

        Args:
            callback (callable): The function to execute upon completion.
        """
        self._on_finish_callback = callback
        
    def __iter__(self):
        try:
            for chunk in self.streamer:
                if isinstance(chunk, str):
                    self._accumulated_text += chunk
                yield chunk
        finally:
            if self._on_finish_callback is not None:
                self._on_finish_callback()
            
    def get_full_result(self) -> str:
        """
        Retrieve the completely accumulated text.

        Returns:
            str: The full generated sentence/text.
        """
        return self._accumulated_text
