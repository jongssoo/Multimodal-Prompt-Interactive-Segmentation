from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from torch import nn

class TextEncoder(nn.Module):
    def __init__(self, clip_ckpt):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_ckpt)
        self.dim_align = nn.Linear(512, 256)
        #self.dim_align = nn.Linear(512, 32)
        #self.device = device
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names,device):
        text_list = ['{}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].to(device)
        return tokens
    
    def forward(self, text, device):
        if text is None:
            return None
        if type(text) is str:
            text = [text]
        tokens = self.organ2tokens(text,device)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)

        return text_embedding