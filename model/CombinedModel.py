import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths):
        img_features = self.encoder(images)
        outputs = self.decoder(img_features, captions, lengths)
        return outputs 