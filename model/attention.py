import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Attention(nn.Module):
    """Attention"""
    def __init__(self, hidden_size, method='general', device='cpu'):
        super(Attention, self).__init__()
        """
        * hidden_size: decoder hidden_size(H_d=encoder_gru_direction*H)
        methods:
        - 'dot': dot product between hidden and encoder_outputs
        - 'general': encoder_outputs through a linear layer 
        - 'concat': concat (hidden, encoder_outputs) ***NOT YET***
        - 'paper': concat + tanh ***NOT YET***
        """
        self.method = method
        self.device = device
        self.hidden_size = hidden_size 
        if self.method == 'general':
            self.linear = nn.Linear(hidden_size, hidden_size)

    #         elif self.method == 'concat':
    #             self.attn = nn.Linear(self.hidden_size*2, self.hidden_size2)
    #             self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size2))
            
    def forward(self, hiddens, enc_outputs, enc_lengths=None, return_weight=False):
        """
        Inputs:
        - hiddens(previous_hiddens): B, 1, H_d
        - enc_outputs(enc_outputs): B, T_e, H_d
        - enc_lengths: real lengths of encoder outputs
        - return_weight = return weights(alphas)
        Outputs:
        - contexts: B, 1, H_d
        - attns: B, 1, T_e
        """
        hid, out = hiddens, enc_outputs
        # Batch(B), Seq_length(T)
        B, T_d, H = hid.size()
        B, T_e, H = out.size()
        
        score = self.get_score(hid, out)
        # score: B, 1, T_e
        if enc_lengths is not None:
            mask = self.get_mask(B, T_d, T_e, enc_lengths)  # masks: B, 1, T_e
            score = score.masked_fill(mask, float('-inf'))
        
        attns = torch.softmax(score, dim=2)  # attns: B, 1, T_e
        contexts = attns.bmm(out)
        if return_weight:
            return contexts, attns
        return contexts
            
    def get_score(self, hid, out):
        """
        Inputs:
        - hid(previous_hiddens): B, 1, H_d 
        - out(enc_outputs): B, T_e, H_d
        Outputs:
        - score: B, 1, T_e
        """
        if self.method == 'dot':
            # bmm: (B, 1, H_d) * (B, H, T_e) = (B, 1, T_e)
            score = hid.bmm(out.transpose(1, 2))
            return score
        
        elif self.method == 'general':
            # linear: (B, T_e, H_d) > (B, T_e, H_d)
            # bmm: (B, 1, H_d) * (B, H_d, T_e) = (B, 1, T_e)
            score = self.linear(out)
            score = hid.bmm(score.transpose(1, 2))
            return score
        
#         elif self.method == 'concat':
#             # H repeat: (B, 1, n_d) > (B, T_x, n_d)
#             # cat: (B, T_x, 2*n_d)
#             # attn: (B, T_x, 2*n_d) > (B, T_x, n_d)
#             # v repeat: (1, n_d) > (B, 1, n_d)
#             # bmm: (B, 1, n_d) * (B, n_d, T_x) = (B, 1, T_x)
#             cat = torch.cat([H.repeat(1, O.size(1), 1), O], 2)
#             e = self.attn(cat)
#             v = self.v.repeat(O.size(0), 1).unsqueeze(1)
#             e = v.bmm(e.transpose(1, 2))
#             return e
        
#         elif self.method == 'paper':
#             # add tanh after attention linear layer in 'concat' method
#             cat = torch.cat([H.repeat(1, O.size(1), 1), O], 2)
#             e = F.tanh(self.attn(cat))
#             v = self.v.repeat(O.size(0), 1).unsqueeze(1)
#             e = v.bmm(e.transpose(1, 2))
#             return e

    def get_mask(self, B, T_d, T_e, lengths):
        assert isinstance(lengths, list), "lengths must be list type"
        mask = torch.zeros(B, T_d, T_e, dtype=torch.uint8).to(self.device)
        for i, x in enumerate(lengths):
            if x < T_e:
                mask[i, :, x:].fill_(1)
        return mask