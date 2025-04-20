import os
import gdown
import torch
import torch

torch.backends.quantized.engine = 'fbgemm' 

def load_vocab():
    if not os.path.exists("vocab.pth"):
        print("Downloading vocab...")
        file_id = "1iedHly1ZGTAK1JnW-vH0NQXUug6TKF2A"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "vocab.pth",quiet=False)
    
    vocab = torch.load("vocab.pth", weights_only=False)
    return vocab

def load_model():
    if not os.path.exists("quantized_model_full.pth"):
        print("Downloading model...")
        file_id = "135yD-wPaoW1R_pPX-4egrA__dzNRRg6P"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "quantized_model_full.pth", quiet=False)

    model = torch.load('quantized_model_full.pth', weights_only=False)
    return model