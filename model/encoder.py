import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layernormGRU import LayerNormGRU

class Encoder(nn.Module):
    """Encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, drop_rate, bidirec=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirec else 1
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=bidirec, 
                          batch_first=True)
        
    def forward(self, inputs, lengths):
        """
        Inputs:
        - inputs: B, T_e
        - lengths: B, (list)
        Outputs:
        - outputs: B, T_e, n_directions*H
        - hiddens: 1, B, n_directions*H
        """
        assert isinstance(lengths, list), "lengths must be a list type"
        # B: batch_size, T_e: enc_length, M: embed_size, H: hidden_size
        inputs = self.embedding(inputs) # (B, T_e) > (B, T_e, m)
        inputs = self.dropout(inputs)
        
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        # packed_inputs: (B*T_e, M) + batches: (T_e)
        packed_outputs, hiddens = self.gru(packed_inputs)
        # packed_outputs: (B*T_e, n_directions*H) + batches: (T_e)
        # hiddens: (n_layers*n_directions, B, H)
        outputs, outputs_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        # output: (B, T_e, H) + lengths (B)
        hiddens = torch.cat([h for h in hiddens[-self.n_direction:]], 1).unsqueeze(0)
        # hiddens: (1, B, n_directions*H)
        return outputs, hiddens