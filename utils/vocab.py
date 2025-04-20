from collections import Counter
import torch
import numpy as np

class Vocabulary:
  def __init__(self,freq_threshold=5):
    self.freq_threshold = freq_threshold
    self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
    self.word2idx = {v:k for k, v in self.idx2word.items()}
    self.embeddings = None
    self.word_freq = Counter()
    self.idx = 4


  def __len__(self):
    return len(self.word2idx)


  def build_vocab(self,captions_dict):
    for _, captions in captions_dict.items():
      for caption in captions:
        self.word_freq.update(caption.split())
    for word,freq in self.word_freq.items():
      if freq >= self.freq_threshold:
        self.idx2word[self.idx] = word
        self.word2idx[word] = self.idx
        self.idx += 1


  def numericalize(self, text):
    tokens = text.split()
    return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]


  def load_glove(self, glovepath, dim = 300):
    embedding_index = {}
    with open(glovepath, 'r', encoding='utf-8') as f:
      for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = vector
    vocab_size = len(self.word2idx)
    embedding_matrix = np.zeros((vocab_size, dim))
    for word, idx in self.word2idx.items():
      if word in embedding_index:
        embedding_matrix[idx] = embedding_index[word]
      else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(dim,))
    self.embeddings = torch.tensor(embedding_matrix, dtype=torch.float32)
