import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab, num_layers=1):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.embed.weight = nn.Parameter(vocab.embeddings, requires_grad=False)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.img_fc = nn.Linear(4096, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, img_features, captions, lengths):
        embeddings = self.dropout(self.embed(captions))
        img_embeds = self.dropout(self.img_fc(img_features)).unsqueeze(1)
        embeddings = torch.cat((img_embeds, embeddings), dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs 