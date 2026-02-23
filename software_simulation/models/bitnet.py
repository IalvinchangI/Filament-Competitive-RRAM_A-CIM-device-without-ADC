import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention, LlamaMLP
import os
from utils import LoggingColor

_logger = LoggingColor.get_logger("bitnet")

def get_TernaryBitNet(checkpoint_path: str, device: str = "cpu") -> LlamaForCausalLM:
    """
    Initialize and load the BitNet b1.58 2B-4T model.

    Performs structural modifications to a standard Llama architecture
    to match the BitNet specifications:
    1. Injects specific Sub-Layer Normalization (SubLN) layers.
    2. Replaces standard Linear layers with BitLinear (1.58-bit).
    3. Replaces standard activations with SquaredReLU.

    Note:
        Pretrained weights for this model can be downloaded from Hugging Face:
        https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
    
    Args:
        checkpoint_path (str): Path to the model weights (.pt, .bin, or .safetensors).
        device (str, optional): Device to load the weights onto (e.g., "cpu", "cuda"). Defaults to "cpu".

    Returns:
        LlamaForCausalLM: The constructed and loaded BitNet model.
    """

    config = LlamaConfig(
        vocab_size=128256, 
        hidden_size=2560, 
        intermediate_size=6912, 
        num_hidden_layers=30, 
        num_attention_heads=20, 
        num_key_value_heads=5, 
        max_position_embeddings=4096, 
        rms_norm_eps=1e-05, 
        rope_theta=500000.0, 
        tie_word_embeddings=True, 
        hidden_act="silu" 
    )

    model = LlamaForCausalLM(config)
    
    _inject_subln_structure(model, config)
    _logger.info(LoggingColor.color_text("SubLN structure injected.", LoggingColor.GREEN))
    
    _replace_linear(model)
    _logger.info(LoggingColor.color_text("Linear layers replaced with BitLinear.", LoggingColor.GREEN))
    
    _replace_activation(model)
    _logger.info(LoggingColor.color_text("Activations replaced with SquaredReLU.", LoggingColor.GREEN))

    if os.path.exists(checkpoint_path):
        _logger.info(f"Loading weights from {checkpoint_path}...")
        
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path, device=device)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            _logger.warning(LoggingColor.color_text("lm_head.weight not found. Cloning from embed_tokens (Weight Tying).", LoggingColor.WARNING))
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict, strict=False)
        _logger.info(LoggingColor.color_text("Pretrained weights loaded successfully!", LoggingColor.BOLD + LoggingColor.GREEN))

    return model

def _inject_subln_structure(model: LlamaForCausalLM, config: LlamaConfig):
    """
    Inject `attn_sub_norm` and `ffn_sub_norm` into Llama layers to match the BitNet structure.
    """
    for layer in model.model.layers:
        layer.self_attn.attn_sub_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer.mlp.ffn_sub_norm = LlamaRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

def _replace_linear(model: LlamaForCausalLM):
    """
    Replace standard Linear layers with BitLinear and assign SubLN references.
    
    Wiring Logic:
    - Q, K, V, Gate, Up: Use standard pre-normalization.
    - O, Down: Use the injected SubLN.
    """
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            attn_norm = module.attn_sub_norm
            
            module.q_proj = BitLinear(module.q_proj.in_features, module.q_proj.out_features, False, sub_norm=None)
            module.k_proj = BitLinear(module.k_proj.in_features, module.k_proj.out_features, False, sub_norm=None)
            module.v_proj = BitLinear(module.v_proj.in_features, module.v_proj.out_features, False, sub_norm=None)
            module.o_proj = BitLinear(module.o_proj.in_features, module.o_proj.out_features, False, sub_norm=attn_norm)

        elif isinstance(module, LlamaMLP):
            ffn_norm = module.ffn_sub_norm
            
            module.gate_proj = BitLinear(module.gate_proj.in_features, module.gate_proj.out_features, False, sub_norm=None)
            module.up_proj = BitLinear(module.up_proj.in_features, module.up_proj.out_features, False, sub_norm=None)
            module.down_proj = BitLinear(module.down_proj.in_features, module.down_proj.out_features, False, sub_norm=ffn_norm)

def _replace_activation(model: LlamaForCausalLM):
    """
    Replace the activation function in MLP layers with SquaredReLU.
    """
    for name, child in model.named_children():
        if "mlp" in name.lower():
            if hasattr(child, "act_fn"):
                child.act_fn = SquaredReLU()
        _replace_activation(child)

class BitLinear(nn.Linear):
    """
    BitLinear (1.58-bit)
    ===========================================================================
    A Linear layer implementation designed for 1.58-bit (ternary) weight quantization 
    and 8-bit input activation quantization.

    Key Responsibilities:
    1. Quantization: Manages the scaling and ternary clamping of weights {-1, 0, 1}, 
       and 8-bit mapping of inputs.
    2. Sub-Layer Normalization: Incorporates optional external RMSNorm processing 
       before quantization.
    3. Pluggable MatMul: Exposes a static method injection point to route matrix 
       multiplication to custom hardware accelerators (e.g., RRAM Driver).
    ===========================================================================
    """

    @staticmethod
    def _default_matmul(x_quant: torch.Tensor, layer_instance: "BitLinear") -> torch.Tensor:
        """Default software implementation using PyTorch FP/INT operations."""
        w_quant = layer_instance.get_quantization_weights()
        y_raw = F.linear(x_quant, w_quant, bias=None)
        return y_raw

    _matmul = _default_matmul
    
    @classmethod
    def set_matmul(cls, function=None):
        """
        Set the global matrix multiplication driver for all BitLinear instances.
        
        Args:
            function (callable, optional): A function (x_quant, layer_id) -> output_tensor. 
                                           If None, reverts to default software simulation.
        """
        if function is None:
            _logger.info(LoggingColor.color_text("[BitLinear] Mode Switch: Software Simulation (Default)", LoggingColor.GREEN))
            cls._matmul = staticmethod(cls._default_matmul)
            return
        
        def hardware_adapter(x_quant, layer_instance):
            if layer_instance.layer_id is None:
                raise RuntimeError("Layer ID not set. Please configure hardware first.")
            return function(x_quant, layer_instance.layer_id)

        _logger.info(LoggingColor.color_text("[BitLinear] Mode Switch: Hardware Acceleration (Driver Mounted)", LoggingColor.GREEN))
        cls._matmul = staticmethod(hardware_adapter)

    def __init__(self, in_features: int, out_features: int, bias: bool = False, sub_norm: nn.Module = None):
        """
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to True, the layer will learn an additive bias. Defaults to False.
            sub_norm (nn.Module, optional): An external normalization instance (like LlamaRMSNorm). 
                                            Defaults to None.
        """
        super(BitLinear, self).__init__(in_features, out_features, bias=False)
        
        self._norm_container = [sub_norm] if sub_norm is not None else []
        self.layer_id: str = None
        self.register_buffer('weight_scale', None)

    def calculate_weight_scale(self):
        """Calculate and store the average absolute value of weights (gamma)."""
        with torch.no_grad():
            self.weight_scale = self.weight.abs().mean().clamp(min=1e-5)
    
    def get_quantization_weights(self) -> torch.Tensor:
        """
        Quantize weights to ternary values {-1, 0, 1}.

        Returns:
            torch.Tensor: The quantized weight tensor.
        """
        if self.weight_scale is None:
             self.calculate_weight_scale()
        return torch.round(self.weight / self.weight_scale).clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying SubLN, input quantization, and matrix multiplication.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The dequantized output tensor.
        """
        if self.weight_scale is None:
            self.calculate_weight_scale()

        if self._norm_container:
            x_norm = self._norm_container[0](x)
        else:
            x_norm = x 
        
        x_absmax = x_norm.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        input_scale = 127.0 / x_absmax
        x_quant = (x_norm * input_scale).round().clamp(-128, 127)

        y_raw = self._matmul(x_quant, self)

        output = y_raw * (self.weight_scale / input_scale)
        
        return output

class SquaredReLU(nn.Module):
    """
    Squared ReLU Activation Function
    ===========================================================================
    Applies the element-wise function: f(x) = max(0, x)^2
    ===========================================================================
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The activated output tensor.
        """
        return F.relu(x).pow(2)
