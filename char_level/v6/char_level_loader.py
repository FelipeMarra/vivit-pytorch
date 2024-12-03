import torch

class CharLoader():
    """
        Loads random samples of chars from txt files (w/ replacement)
    """
    def __init__(self, path, batch_size, context_size):
        self.path = path
        self.batch_size = batch_size
        self.context_size = context_size

        self.text, self.chars = self.get_text()
        self.vocab_size = len(self.chars)

        print("chars:", ''.join(self.chars))
        print("vocab size:", self.vocab_size)

        self.char2idx, self.idx2char = self.get_vocab(self.chars)
        self.train_data, self.val_data = self.get_split(self.text)

    def get_text(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))

        return text, chars

    def get_vocab(self, chars):
        char2idx = {ch:i for i,ch in enumerate(chars)}
        idx2char = {i:ch for i,ch in enumerate(chars)}

        return char2idx, idx2char

    def encode(self, string):
        encoded = [self.char2idx[c] for c in string] # encoder: take a string, output a list of integers
        return encoded

    def decode(self, encoded_list:list|torch.Tensor):
        encoded_list = encoded_list.tolist() if type(encoded_list) == torch.Tensor else encoded_list

        decoded = []
        for enc_l in encoded_list:
            decoded.append(''.join([self.idx2char[i] for i in enc_l])) # decoder: take a list of integers, output a string

        return decoded

    def get_split(self, text):
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(data)) # first 90% will be train, rest val

        return data[:n], data[n:] #train, val

    def get_batch(self, train=True):
        data = self.train_data if train else self.val_data
        indexes = torch.randint(low=0, high=len(data) - self.context_size, size=(self.batch_size,))

        x = torch.stack([data[i:i+self.context_size] for i in indexes])
        y = torch.stack([data[i+1:i+self.context_size+1] for i in indexes])

        return x.cuda(), y.cuda()