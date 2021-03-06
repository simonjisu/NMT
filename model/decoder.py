import torch
import torch.nn as nn
from attention import Attention

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, sos_idx=2, drop_rate=0.0, layernorm=False, method='general', teacher_force=False, device='cpu', return_w=False):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.sos_idx = sos_idx
        self.return_w = return_w
        self.teacher_force = teacher_force
        self.layernorm = layernorm
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(drop_rate)
        self.attention = Attention(hidden_size, method=method, device=device)
        self.gru = nn.GRU(embed_size+hidden_size, hidden_size, n_layers, bidirectional=False, 
                          batch_first=True)
        self.linear = nn.Linear(2*hidden_size, vocab_size)
        if layernorm:
            self.l_norm = nn.LayerNorm(embed_size)
    
    def start_token(self, batch_size):
        sos = torch.LongTensor([self.sos_idx]*batch_size).unsqueeze(1).to(self.device)
        return sos
    
    def init_hiddens(self, batch_size):
        return torch.zeros(batch_size, self.n_layers, self.hidden_size).to(self.device)
    
    def forward(self, hiddens, enc_output, enc_lengths=None, max_len=None, targets=None, 
                is_eval=False, is_test=False, stop_idx=3):
        """
        * H_d: decoder hidden_size = encoder_gru_directions * H
        * M_d: decoder embedding_size
        Inputs:
        - hiddens: last encoder hidden at time 0 = 1, B, H_d 
        - enc_output: encoder output = B, T_e, H_d
        - enc_lengths: encoder lengths = B
        - max_len: max lenghts of target = T_d
        Outputs:
        - scores: results of all predictions = B*T_d, vocab_size
        - attn_weights: attention weight for all batches = B, T_d, T_e
        """
        if is_test:
            is_eval=True
        inputs = self.start_token(hiddens.size(1))  # (B, 1)
        inputs = self.embedding(inputs) # (B, 1, M_d)
        if self.layernorm:
            inputs = self.l_norm(inputs)
        inputs = self.dropout(inputs)
        # match layer size: (1, B, H_d) > (n_layers, B, H_d)
        if hiddens.size(0) != self.n_layers:
            hiddens = hiddens.repeat(self.n_layers, 1, 1)
        # prepare for whole target sentence scores
        scores = []  
        attn_weights = []
        for i in range(1, max_len):
            # contexts[c{i}] = alpha(hiddens[s{i-1}], encoder_output[h])
            # select last hidden: (1, B, H_d) > transpose: (B, 1, H_d) > attention
#             contexts = self.attention(hiddens.transpose(0, 1), enc_output, enc_lengths, 
#                                       return_weight=self.return_w)
            contexts = self.attention(hiddens[-1:, :].transpose(0, 1), enc_output, enc_lengths, 
                                      return_weight=self.return_w)
    
            if self.return_w:
                attns = contexts[1]  # attns: (B, seq_len=1, T_e) 
                contexts = contexts[0]  # contexts: (B, seq_len=1, H_d)
                attn_weights.append(attns)
                
            # gru_inputs = concat(embeded_token[y{i-1}], contexts[c{i}]): (B, seq_len=1, H_d+M_d)
            gru_inputs = torch.cat((inputs, contexts), 2)
            
            # gru: s{i} = f(gru_inputs, s{i-1})
            # (B, 1, M_d+H_d) > (n_layers, B, H_d)
            _, hiddens = self.gru(gru_inputs, hiddens)            
            
            # scores = g(s{i}, c{i})
            # select last hidden: (1, B, H_d) > transpose: (B, 1, H_d) > concat: (B, 1, H_d + H_d) >
            # output linear : (B, seq_len=1, vocab_size)
#             score = self.linear(torch.cat((hiddens.transpose(0, 1), contexts), 2))
            score = self.linear(torch.cat((hiddens[-1:, :].transpose(0, 1), contexts), 2))
            scores.append(score)
    
            if (self.teacher_force and not is_eval):
                selected_targets = targets[:, i].unsqueeze(1)
            else:
                selected_targets = None
            
            inputs, stop_decode = self.decode(is_tf=self.teacher_force, 
                                              is_eval=is_eval,
                                              is_test=is_test,
                                              score=score, 
                                              targets=selected_targets, 
                                              stop_idx=stop_idx)
            if stop_decode:
                break
            
        scores = torch.cat(scores, 1).view(-1, self.vocab_size)  # (B, T_d, vocab_size) > (B*T_d, vocab_size)
        if self.return_w:
            return scores, torch.cat(attn_weights, 1)  # (B, T_d, T_e)
        return scores
    
    def decode(self, is_tf, is_eval, is_test, score, targets, stop_idx):
        """
        - for validation: if is_tf, set 'is_eval' True, else False
        - for test evaluation: set 'is_tf' False and set 'is_eval' True
        """
        stop_decode = False
        if is_test:
            # test
            preds = score.max(2)[1]
            if preds.view(-1).item() == stop_idx:
                stop_decode = True
            inputs = self.embedding(preds)
        else:
            # train & valid
            if is_tf and not is_eval:
                assert targets is not None, "target must not be None in teacher force mode"
                inputs = self.embedding(targets)
            else:
                preds = score.max(2)[1]
                inputs = self.embedding(preds)

        if self.layernorm:
            inputs = self.l_norm(inputs)
        inputs = self.dropout(inputs)
        return inputs, stop_decode