import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from attention import Attention
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.cuda.current_device()

class Decoder(nn.Module):
    def __init__(self, V_d, m_d, n_d, sos_idx, num_layers=1, hidden_size2=None, method='general', 
                 return_weight=True):
        super(Decoder, self).__init__()
        """
        vocab_size: V_d
        embed_size: m_d
        hidden_size: n_d (set this value as 2*n_e)
        methods:
        - 'dot': dot product between hidden and encoder_outputs
        - 'general': encoder_outputs through a linear layer 
        - 'concat': concat (hidden, encoder_outputs)
        - 'paper': concat + tanh
        return_weight: return attention weights
        """
        self.V_d = V_d
        self.m_d = m_d
        self.n_d = n_d
        self.sos_idx = sos_idx
        self.num_layers = num_layers
        self.return_weight = return_weight
        self.method = method
        # attention
        self.attention = Attention(hidden_size=n_d, hidden_size2=hidden_size2, method=method)
        # embed
        self.embed = nn.Embedding(V_d, m_d)
        # gru(W*[embed, context] + U*[hidden_prev])
        # gru: m+n
        self.gru = nn.GRU(m_d+n_d, n_d, num_layers, batch_first=True, bidirectional=False) 
        # linear
        self.linear = nn.Linear(2*n_d, V_d)
        
        
    def start_token(self, batch_size):
        sos = torch.LongTensor([self.sos_idx]*batch_size).unsqueeze(1)
        if USE_CUDA: sos = sos.cuda()
        return sos
    
    def forward(self, hidden, enc_outputs, enc_outputs_lengths, max_len):
        """
        input:
        - hidden(previous hidden): B, 1, n_d 
        - enc_outputs(source context): B, T_x, n_d
        - enc_outputs_lengths: list type
        - max_len(targer sentences max len in batch): T_y
        """
        assert isinstance(enc_outputs_lengths, list), \
            'encoder_outputs_lengths must be a list type, current is {}'.format(type(enc_outputs_lengths))
        
        inputs = self.start_token(hidden.size(0)) # (B, 1)
        embeded = self.embed(inputs) # (B, 1, m_d)
        # prepare for whole targer sentence scores
        scores = []
        attn_weights = []
        
        for i in range(max_len):
            # context vector: previous hidden(s{i-1}), encoder_outputs(O_e) > context(c{i}), weights
            # - context: (B, 1, n_d)
            # - weights: (B, 1, T_x)
            context, weights = self.attention(hidden, enc_outputs, \
                                              enc_outputs_lengths, return_weight=self.return_weight)
            attn_weights.append(weights.squeeze(1))
            
            # concat context & embedding vectors: (B, 1, m_d+n_d)
            gru_input = torch.cat([embeded, context], 2)
            
            # gru((context&embedding), previous hidden)
            # output hidden(s{i}): (1, B, n_d)
            _, hidden = self.gru(gru_input, hidden.transpose(0, 1))
            hidden = hidden.transpose(0, 1) # change shape to (B, 1, n_d) again
            
            # concat context and new hidden vectors: (B, 1, 2*n_d)
            concated = torch.cat([hidden, context], 2)
            
            # get score: (B, V_d)
            score = self.linear(concated.squeeze(1))
            scores.append(score)
            
            # greedy method
            decoded = score.max(1)[1]  # (B)
            embeded = self.embed(decoded).unsqueeze(1) # next input y{i-1} (B, 1, m_d) 

        # column-wise concat, reshape!! 
        # scores = [(B, V_d), (B, V_d), (B, V_d)...] > (B, V_d*max_len)
        # attn_weights = [(B, T_x), (B, T_x), (B, T_x)...] > (B*max_len, T_x)
        scores = torch.cat(scores, 1)
        return scores.view(inputs.size(0)*max_len, -1), torch.cat(attn_weights)