import torch
import torch.nn.functional as F
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        mlp_hidden_sizes: list = [128],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(
            embeddings.shape[1], lstm_hidden_size, batch_first=True, dropout=dropout, num_layers=lstm_num_layers
        )

        # Initialize the MLP input and hidden layers dynamically from the mlp_hidden_sizes array
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(lstm_hidden_size, mlp_hidden_sizes[0]))
        for i in range(len(mlp_hidden_sizes) - 1):
            self.mlp.append(nn.Linear(mlp_hidden_sizes[i], mlp_hidden_sizes[i + 1]))
        # Initialize the output layer
        self.mlp.append(nn.Linear(mlp_hidden_sizes[-1], 1))

    def forward(self, x):
        seq, mask, lengths = x[0], x[1], x[2]

        emb = self.embedding(seq)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, (hidden, _) = self.lstm(packed_embedded)
        # ignore the output length of the sequences from unpacking
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = F.relu(self.mlp[0](hidden[-1]))

        for layer in self.mlp[1:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout)

        x = self.mlp[-1](x)
        return x

    @property
    def name(self):
        return self.__class__.__name__
