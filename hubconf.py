import json
import torch
from redimnet import ReDimNetWrap
dependencies = ['torch','torchaudio']

URL_TEMPLATE = "https://github.com/IDRnD/ReDimNet/releases/download/latest/{model_name}"
 
def load_custom(size, pretrained=False, finetuned=True):
    model_prefix = 'lm' if finetuned else 'ptn'
    assert size in [f'b{i}' for i in range(7)]

    model_name = f'redimnet-{size}-vox2-{model_prefix}.pt'
    url = URL_TEMPLATE.format(model_name = model_name)
    
    full_state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    
    model_config = full_state_dict['model_config']
    state_dict = full_state_dict['state_dict']
    model = ReDimNetWrap(**model_config)
    if pretrained or finetuned:
        load_res = model.load_state_dict(state_dict)    
        print(f"load_res : {load_res}")
    return model

def b0(pretrained=False, finetuned=True):
    return load_custom('b0', pretrained=pretrained, finetuned=finetuned)

def b1(pretrained=False, finetuned=True):
    return load_custom('b1', pretrained=pretrained, finetuned=finetuned)

def b2(pretrained=False, finetuned=True):
    return load_custom('b2', pretrained=pretrained, finetuned=finetuned)

def b3(pretrained=False, finetuned=True):
    return load_custom('b3', pretrained=pretrained, finetuned=finetuned)

def b5(pretrained=False, finetuned=True):
    return load_custom('b5', pretrained=pretrained, finetuned=finetuned)

def b6(pretrained=False, finetuned=True):
    return load_custom('b6', pretrained=pretrained, finetuned=finetuned)