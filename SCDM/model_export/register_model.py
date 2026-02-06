import torch
import torchvision
import torch.nn as nn
from model_export import ModelImportHandler
from transformers import AutoConfig, AutoModelForCausalLM

from third_party_models import reactnet
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.model.sew_resnet import sew_resnet34 as sew_resnet
from my_token import HP_TOKEN

# ==============================================================================
# [Vision] BNN: ReActNet-A
# ==============================================================================
@ModelImportHandler.register_model("Vision_ReActNet_BNN")
def import_vision_bnn():
    print(f"\n[Vision] Initializing ReActNet-A (BNN)...")

    weights_path = ModelImportHandler.MODEL_DEFAULT_DIR / "ReActNet_A.pth.tar"
    
    try:
        print(" -> Found 'reactnet.py'. Initializing ReActNet-A structure...")
        model = reactnet()
        
        if weights_path.exists():
            print(f" -> Loading real weights from {weights_path.name}...")
            checkpoint = torch.load(weights_path, map_location=ModelImportHandler.DEVICE)
            
            if 'state_dict' in checkpoint:
                raw_state_dict = checkpoint['state_dict']
            else:
                raw_state_dict = checkpoint
            
            # 1. fix the difference of weight and weights
            # 2. fix shape
            model_state_dict = model.state_dict()
            new_state_dict = {}
            for k, v in raw_state_dict.items():
                k = k.replace('module.', '')
                
                # 1. fix the layers whose name include 'binary'
                if 'binary' in k and k.endswith('.weight'):
                    # .weight -> .weights
                    new_key = k.replace('.weight', '.weights')
                    new_state_dict[new_key] = v
                    
                    # 2. fix shape
                    if new_key in model_state_dict:
                        target_shape = model_state_dict[new_key].shape
                        
                        if v.shape != target_shape and v.numel() == target_shape.numel():
                            v = v.view(target_shape)
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict)
            print(" -> [Success] Real ReActNet-A loaded!")
            
            model.to(ModelImportHandler.DEVICE)
            return model
        else:
            print(f" -> Weights not found at {weights_path}.")
            return None

    except ImportError:
        print(" -> [Info] 'reactnet.py' not found.")
    except Exception as e:
        print(f" -> [Error] loading ReActNet: {e}.")
    return None

# ==============================================================================
# [Vision] SNN: SEW-ResNet
# ==============================================================================
@ModelImportHandler.register_model("Vision_SEWResNet34_SNN")
def import_vision_snn():
    print(f"\n[Vision] Initializing SEW-ResNet (SNN)...")
    try:
        try:
            model = sew_resnet(
                pretrained=True, 
                spiking_neuron=neuron.IFNode,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
            print(" -> Loaded Pre-trained SEW-ResNet weights.")
        except:
            model = sew_resnet(
                pretrained=False, 
                spiking_neuron=neuron.IFNode,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
            print(" -> Loaded Random SEW-ResNet structure.")
            
        model.to(ModelImportHandler.DEVICE)
        return model
    except Exception as e:
        print(f" -> [Error] loading SEW-ResNet34: {e}.")
        return None

# ==============================================================================
# [NLP] TNN: BitNet b1.58
# ==============================================================================
@ModelImportHandler.register_model("NLP_BitNet_0.7B_TNN")
def import_nlp_tnn():
    print(f"\n[NLP] Initializing BitNet b1.58 (TNN)...")
    try:
        model_id = "1bitLLM/bitnet_b1_58-large"
        
        try:
            print(f" -> Attempting to download REAL weights from {model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto", 
                dtype=torch.float16,
                low_cpu_mem_usage=True, 
                token=HP_TOKEN
            )
            print(" -> Successfully loaded REAL BitNet weights.")
        except Exception as e:
            print(f" -> Download failed ({e}). Using TinyLlama Proxy structure.")
            config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            model = AutoModelForCausalLM.from_config(config)
            
            print(" -> Applying Fake TNN Quantization (-1, 0, +1)...")
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        threshold = 0.5 * param.data.abs().mean()
                        param.data = torch.where(param.data > threshold, torch.tensor(1.0),
                                     torch.where(param.data < -threshold, torch.tensor(-1.0), torch.tensor(0.0)))
        
        return model

    except ImportError:
        print(" -> [Error] 'transformers' not installed. Skipping NLP.")
        return None

if __name__ == "__main__":
    ModelImportHandler.import_all_model()
