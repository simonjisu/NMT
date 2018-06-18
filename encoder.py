import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.cuda.current_device()

class Encoder(nn.Module):
    def __init__(self, V_e, m_e, n_e, num_layers=1, bidrec=False):
        super(Encoder, self).__init__()
        """
        vocab_size: V_e
        embed_size: m_e
        hidden_size: n_e
        """
        self.V_e = V_e
        self.m_e = m_e
        self.n_e = n_e
        self.num_layers = num_layers
        self.bidrec = bidrec
        self.n_direct = 2 if bidrec else 1
        
        self.embed = nn.Embedding(V_e, m_e) 
        self.gru = nn.GRU(m_e, n_e, num_layers, batch_first=True, bidirectional=bidrec)
        
    def forward(self, inputs, lengths):
        """
        input: 
        - inputs: B, T_x
        - lengths: actual max length of batches
        output:
        - outputs: B, T_x, n_e
        """
        # embeded: (B, T_x, n_e)
        embeded = self.embed(inputs) 
        # packed: (B*T_x, n_e)
        packed = pack_padded_sequence(embeded, lengths, batch_first=True) 
        # packed outputs: (B*T_x, 2*n_e)
        # hidden: (num of layers*n_direct, B, 2*n_e)
        outputs, hidden = self.gru(packed)
        # unpacked outputs: (B, T_x, 2*n_e)
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=True)
        
        # hidden bidirection: (num of layers*n_direct(0,1,2...last one), B, n_e)
        # choosen last hidden: (B, 1, 2*n_e)
        hidden = torch.cat([h for h in hidden[-self.n_direct:]], 1).unsqueeze(1)
        
        return outputs, hidden