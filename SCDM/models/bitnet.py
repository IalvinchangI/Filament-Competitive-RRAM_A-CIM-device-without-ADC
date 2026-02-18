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
    Initializes and loads the BitNet b1.58 2B-4T model.

    This function performs structural modifications to a standard Llama architecture
    to match the BitNet specifications, including:
    1. Injecting specific Sub-Layer Normalization (SubLN) layers.
    2. Replacing standard Linear layers with BitLinear (1.58-bit).
    3. Replacing activations with SquaredReLU.

    Args:
        checkpoint_path (str): Path to the model weights (.pt, .bin, or .safetensors).
        device (str): Device to load the weights onto (e.g., "cpu", "cuda").

    Returns:
        LlamaForCausalLM: The constructed BitNet model.
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
    
    # 1. Structural Modification: Inject SubLN layers
    # This aligns the model structure with the checkpoints provided by Microsoft.
    _inject_subln_structure(model, config)
    _logger.info(LoggingColor.color_text("SubLN structure injected.", LoggingColor.GREEN))
    
    # 2. Linear Layer Replacement
    # Replaces nn.Linear with BitLinear and wires the specific SubLN to the correct layers.
    _replace_linear(model)
    _logger.info(LoggingColor.color_text("Linear layers replaced with BitLinear.", LoggingColor.GREEN))
    
    # 3. Activation Replacement
    _replace_activation(model)
    _logger.info(LoggingColor.color_text("Activations replaced with SquaredReLU.", LoggingColor.GREEN))

    # 4. Load Weights
    if os.path.exists(checkpoint_path):
        _logger.info(f"Loading weights from {checkpoint_path}...")
        
        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path, device=device)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle missing lm_head weight due to weight tying
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            _logger.warning(LoggingColor.color_text("lm_head.weight not found. Cloning from embed_tokens (Weight Tying).", LoggingColor.WARNING))
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict, strict=False)
        _logger.info(LoggingColor.color_text("Pretrained weights loaded successfully!", LoggingColor.BOLD + LoggingColor.GREEN))

    return model

def _inject_subln_structure(model: LlamaForCausalLM, config):
    """
    Injects `attn_sub_norm` and `ffn_sub_norm` into Llama layers to match BitNet checkpoint structure.
    """
    for layer in model.model.layers:
        # SubLN for Attention Output (O_proj)
        layer.self_attn.attn_sub_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # SubLN for MLP Output (Down_proj)
        # Note: Dimension matches intermediate_size (6912)
        layer.mlp.ffn_sub_norm = LlamaRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

def _replace_linear(model: LlamaForCausalLM):
    """
    Replaces standard Linear layers with BitLinear and assigns SubLN references.
    
    Wiring Logic:
    - Q, K, V, Gate, Up: Use standard pre-normalization (input_layernorm/post_attention_layernorm).
    - O, Down: Use the injected SubLN (attn_sub_norm/ffn_sub_norm).
    """
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            attn_norm = module.attn_sub_norm
            
            # Q, K, V: No extra SubLN (already normalized by input_layernorm)
            module.q_proj = BitLinear(module.q_proj.in_features, module.q_proj.out_features, False, sub_norm=None)
            module.k_proj = BitLinear(module.k_proj.in_features, module.k_proj.out_features, False, sub_norm=None)
            module.v_proj = BitLinear(module.v_proj.in_features, module.v_proj.out_features, False, sub_norm=None)
            
            # O_proj: Uses attn_sub_norm
            module.o_proj = BitLinear(module.o_proj.in_features, module.o_proj.out_features, False, sub_norm=attn_norm)

        elif isinstance(module, LlamaMLP):
            ffn_norm = module.ffn_sub_norm
            
            # Gate, Up: No extra SubLN (already normalized by post_attention_layernorm)
            module.gate_proj = BitLinear(module.gate_proj.in_features, module.gate_proj.out_features, False, sub_norm=None)
            module.up_proj = BitLinear(module.up_proj.in_features, module.up_proj.out_features, False, sub_norm=None)
            
            # Down_proj: Uses ffn_sub_norm
            module.down_proj = BitLinear(module.down_proj.in_features, module.down_proj.out_features, False, sub_norm=ffn_norm)

def _replace_activation(model: LlamaForCausalLM):
    """
    Replaces the activation function in MLP layers with SquaredReLU.
    """
    for name, child in model.named_children():
        if "mlp" in name.lower():
            if hasattr(child, "act_fn"):
                child.act_fn = SquaredReLU()
        _replace_activation(child)

class BitLinear(nn.Linear):
    """
    A Linear layer implementing 1.58-bit quantization (ternary weights) and 8-bit activation quantization.
    Supports pluggable hardware drivers for matrix multiplication.
    """

    @staticmethod
    def _default_matmul(x_quant, layer_instance: "BitLinear"):
        """Default software implementation using PyTorch FP/INT operations."""
        w_quant = layer_instance.get_quantization_weights()
        # Simulated linear operation using quantized values
        y_raw = F.linear(x_quant, w_quant, bias=None)
        return y_raw

    _matmul = _default_matmul
    
    @classmethod
    def set_matmul(cls, function=None):
        """
        Sets the global matrix multiplication driver.
        
        Args:
            function: A callable (x_quant, layer_id) -> output_tensor. 
                      If None, reverts to default software simulation.
        """
        if function is None:
            print(" -> [BitLinear] Mode Switch: Software Simulation (Default)")
            cls._matmul = cls._default_matmul
            return
        
        def hardware_adapter(x_quant, layer_instance):
            if layer_instance.layer_id is None:
                raise RuntimeError("Layer ID not set. Please configure hardware first.")
            return function(x_quant, layer_instance.layer_id)

        print(" -> [BitLinear] Mode Switch: Hardware Acceleration (Driver Mounted)")
        cls._matmul = hardware_adapter

    def __init__(self, in_features, out_features, bias=False, sub_norm=None):
        """
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
            sub_norm: Optional LlamaRMSNorm instance for Sub-Layer Normalization.
        """
        super(BitLinear, self).__init__(in_features, out_features, bias=False)
        
        # Use a list container to hold the reference to the external Norm layer.
        # This prevents PyTorch from registering it as a child parameter of this layer,
        # avoiding 'missing key' errors during state_dict loading.
        self._norm_container = [sub_norm] if sub_norm is not None else []
        
        self.layer_id: str = None
        self.register_buffer('weight_scale', None)

    def calculate_weight_scale(self):
        """Calculates the average absolute value of weights (gamma)."""
        with torch.no_grad():
            self.weight_scale = self.weight.abs().mean().clamp(min=1e-5)
    
    def get_quantization_weights(self):
        """Quantizes weights to ternary values {-1, 0, 1}."""
        if self.weight_scale is None:
             self.calculate_weight_scale()
        return torch.round(self.weight / self.weight_scale).clamp(-1, 1)

    def forward(self, x):
        if self.weight_scale is None:
            self.calculate_weight_scale()

        # 1. SubLN (Reference Call)
        # Apply external normalization if injected (e.g., for O_proj and Down_proj)
        if self._norm_container:
            x_norm = self._norm_container[0](x)
        else:
            x_norm = x 
        
        # 2. Input Quantization (Absmax)
        x_absmax = x_norm.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        input_scale = 127.0 / x_absmax
        x_quant = (x_norm * input_scale).round().clamp(-128, 127)

        # 3. Matrix Multiplication (Driver-dependent)
        y_raw = self._matmul(x_quant, self)

        # 4. Dequantization
        # Rescale the integer output back to floating point
        output = y_raw * (self.weight_scale / input_scale)
        
        return output

class SquaredReLU(nn.Module):
    """
    Squared ReLU activation function: f(x) = max(0, x)^2
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.relu(x).pow(2)
