import torch
import numpy
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
import traceback
import gc

from model_loading import BasicModelLoader
from models import BitLinear
from utils import LoggingColor
from utils.environment_variables import HP_TOKEN, TORCH_DEVICE

from torch import nn
from typing import Union, Iterator
from SCDM_Driver import SCDM_DriverInterface


class TernaryBitNetLoader(BasicModelLoader):

    BIT_DEPTH = 8 # 硬體輸入位元深度 (x_quant 是 int8)
    MAX_TOKEN = 50
    FIX_OUTPUT_GENERATION = False
    STREAM_GENERATE_TIMEOUT = None  # no timeout
    # STREAM_GENERATE_TIMEOUT = 20.0
    
    @classmethod
    def set_MAX_TOKEN(cls, value: int):
        if value <= 0:
            raise ValueError("MAX_TOKEN should greater than 0.")
        cls.MAX_TOKEN = value
    
    # Class Level 的 Tokenizer 快取
    _tokenizer = None
    _DEFAULT_TOKENIZER_ID = "microsoft/bitnet-b1.58-2B-4T"

    @classmethod
    def init_tokenizer(cls, id: str = None) -> bool:
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
            cls._logger.error(LoggingColor.color_text(f"❌ Failed to load tokenizer: {e}", LoggingColor.ERROR))
            return False

    GENERATE_COMMAND = "generate_command"
    INFERENCE_COMMAND = "inference_command"
    STREAM_GENERATE_COMMAND = "stream_generate_command"

    _logger = LoggingColor.get_logger("TernaryBitNetLoader")

    @classmethod
    def str2token(cls, string: str) -> numpy.ndarray:
        """將字串轉換為 Token ID (Numpy Array)"""
        if cls._tokenizer is None:
            if cls.init_tokenizer() == False:
                return numpy.array([])

        encoded = cls._tokenizer(string, return_tensors="np")
        return encoded.input_ids

    @classmethod
    def token2str(cls, tokens: numpy.ndarray) -> str:
        """
        將 Token ID 轉換為字串
        支援輸入: 單一整數, List, Numpy Array, Torch Tensor
        """
        if cls._tokenizer is None:
            if cls.init_tokenizer() == False:
                return ""

        # 1. 型別標準化 (Normalize to int or List[int])
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
            
        # 2. 解碼
        decoded = cls._tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded

    @classmethod
    def print_stream(cls, stream_result: Iterator[str]) -> Iterator[str]:
        """
        直接印出 stream 的東西
        
        Usage:
            raw_stream = loader.run(..., cmd=STREAM_GENERATE_COMMAND)
            BitNetLoader.print_stream(raw_stream)
        """
        if stream_result is None:
            return

        try:
            for new_text in stream_result:
                print(new_text, end="", flush=True)
        except Exception as e:
            # TODO stop thread
            cls._logger.error(f"Stream decoding error: {e}")

    def __init__(self, model_path: str):
        super().__init__()
        # init model
        try:
            self._logger.info(f"⏳ Loading model from {model_path}")
            self.model: nn.Module = torch.load(model_path, map_location=TORCH_DEVICE, weights_only=False)
            self.model.eval()
            
            # 自動偵測 device
            self.device = next(self.model.parameters()).device
            
            self._logger.info(LoggingColor.color_text("✅ Loaded.", LoggingColor.GREEN))
        except Exception as e:
            self._logger.error(LoggingColor.color_text(f"❌ Load failed: {e}", LoggingColor.ERROR))
            self.model = None

        # init driver
        self.driver: SCDM_DriverInterface = None

    def config(self, driver: SCDM_DriverInterface):
        """配置硬體加速"""
        super().config(driver)
        
        if self.model is None:
            return

        self._logger.info("⚙️ Configuring BitNet hardware acceleration...")
        count = 0

        # submit weights
        for name, module in self.model.named_modules():
            if isinstance(module, BitLinear):
                # 1. 確保 Scale 已計算
                module.calculate_weight_scale()

                # 2. 取得量化權重
                w_quant_tensor = module.get_quantization_weights().to(torch.int8)
                w_quant_np = w_quant_tensor.cpu().numpy().T
                
                # 3. Setup Hardware
                gid = driver.submit_matrix([w_quant_np])
                
                # 4. 綁定 ID
                module.layer_id = gid
                count += 1

                # 5. 刪除 PyTorch 原本的權重參數，釋放記憶體
                del module.weight
        
        # 強制進行垃圾回收並清空 GPU 快取
        gc.collect()
        torch.cuda.empty_cache()
        
        self._logger.info(f"✅ Configured {count} BitLinear layers to Hardware.")

        # Define Adapter
        def _matmul_hardware_adapter(x_quant_tensor, layer_id: str):
            x_np = x_quant_tensor.cpu().numpy().astype(numpy.int8)
            y_np = driver.compute_multibit(layer_id, x_np, self.BIT_DEPTH)
            y_tensor = torch.from_numpy(y_np).to(device=x_quant_tensor.device)
            return y_tensor

        # Global Switch
        BitLinear.set_matmul(_matmul_hardware_adapter)

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> Union[numpy.ndarray, Iterator[str]]:
        """
        執行模型
        Returns:
            numpy.ndarray: 若為 INFERENCE 或 GENERATE
            Iterator[str]: 若為 STREAM_GENERATE (回傳 Python Generator)
        """
        super().run(input_data, input_command)

        if self.model is None:
            self._logger.error(LoggingColor.color_text("❌ Model not loaded.", LoggingColor.ERROR))
            return None
        
        # Prepare Input
        input_tensor = torch.from_numpy(input_data).to(self.device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        attention_mask = torch.ones_like(input_tensor)  # WARMING: The attention_mask is hard-coded as an all-ones array.

        self._logger.info(f"🚀 Running {input_command}...")

        output_data = None

        try:
            if input_command == self.INFERENCE_COMMAND or input_command is None:
                # 單次推論
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits
                output_data = logits.cpu().numpy()
            
            elif input_command == self.GENERATE_COMMAND:
                # 一般生成 (Blocking)
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
                # 串流生成 (Non-Blocking / Generator)
                # 1. 確保 Tokenizer 載入 (Streamer 需要它來解碼)
                if self._tokenizer is None:
                    if not self.init_tokenizer():
                         return None
                
                # 2. 建立 Streamer (這是一個 Iterator)
                # skip_prompt=True: 不重複回傳輸入的字
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=self.STREAM_GENERATE_TIMEOUT)

                # 3. 設定生成參數
                generation_kwargs = dict(
                    inputs=input_tensor, 
                    attention_mask=attention_mask, 
                    streamer=streamer, 
                    max_new_tokens=self.MAX_TOKEN, 
                    pad_token_id=self.model.config.eos_token_id, 
                    do_sample=(not self.FIX_OUTPUT_GENERATION)
                )

                # 4. 在子執行緒中啟動生成
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                # 5. 直接回傳 Streamer (Generator)
                # 外部可以用 `for text in result: print(text)` 來接收
                return TextStreamTracker(streamer)

            else:
                self._logger.warning(LoggingColor.color_text(f"⚠️ Unknown command: {input_command}", LoggingColor.WARNING))
                return None

        except Exception as e:
            self._logger.error(LoggingColor.color_text(f"❌ Runtime error: {e}", LoggingColor.ERROR))
            traceback.print_exc()
            return None

        return output_data
        
    def unload(self):
        super().unload()
        if self.model is None or self.driver is None:
            return

        self._logger.info("🧹 Unloading model resources...")
        count = 0
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                if module.layer_id is not None:
                    if self.driver.clear_matrix(module.layer_id):
                        count += 1
                    module.layer_id = None 

        BitLinear.set_matmul(None) 
        self._logger.info(f"✅ Cleared {count} matrices from hardware.")

    def loader_details(self):
        return {
            "bit_depth": self.BIT_DEPTH, 
            "fix_output_generation": self.FIX_OUTPUT_GENERATION, 
            "max_token": self.MAX_TOKEN
        }



class TextStreamTracker:
    """
    用來包裝 TextIteratorStreamer。
    當前端 (main.py) 用 for 迴圈讀取並印出字串時，
    它會偷偷在背景把完整的句子存起來，供後續 Executor 存檔。
    """
    def __init__(self, streamer):
        self.streamer = streamer
        self._accumulated_text = ""
        self._on_finish_callback = None
    
    def register_on_finish_callback(self, callback):
        """允許外部註冊一個在串流結束時觸發的函式"""
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
        return self._accumulated_text
