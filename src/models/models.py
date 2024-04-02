import torch
from torch import nn
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaModel

class LLAMA_EFR(nn.Module):
    def __init__(self, model_path, hid_dim=3200):
        self.core = LlamaModel.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto', offload_folder="offload_base", offload_state_dict=True)
        
        self.trig_score1 = nn.Linear(hid_dim, hid_dim, bias=True, dtype=torch.float16).to("cuda")
        self.trig_score2 = nn.Linear(hid_dim, 1, bias=False, dtype=torch.float16).to("cuda")
    
    def forward(self, d_ids):
        pass