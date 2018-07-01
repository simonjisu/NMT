import torch
import torch.nn as nn
from attention import Attention

class Decoder(nn.Module):
    def __init__(self, V_d, m_d, n_d, sos_idx=2, num_layers=1, hidden_size2=None, decode_method='greedy',
                 method='general', ktop=5, return_weight=True, max_len=15, dropout_rate=0.0, USE_CUDA=True):
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
        self.dec_method = decode_method
        self.ktop = ktop
        self.use_dropout = False if dropout_rate == 0.0 else True
        self.USE_CUDA = USE_CUDA
        # attention
        self.attention = Attention(hidden_size=n_d, hidden_size2=hidden_size2, method=method)
        # embed
        self.embed = nn.Embedding(V_d, m_d)
        # dropout:
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        # gru(W*[embed, context] + U*[hidden_prev])
        # gru: m+n
        self.gru = nn.GRU(m_d+n_d, n_d, num_layers, batch_first=True, bidirectional=False)
        # linear
        self.linear = nn.Linear(2*n_d, V_d)
        self.max_len = max_len
        
        
    def start_token(self, batch_size):
        sos = torch.LongTensor([self.sos_idx]*batch_size).unsqueeze(1)
        if self.USE_CUDA: sos = sos.cuda()
        return sos
    
    def forward(self, hidden, enc_outputs, enc_outputs_lengths=None, max_len=None):
        """
        input:
        - hidden(previous hidden): B, 1, n_d 
        - enc_outputs(source context): B, T_x, n_d
        - enc_outputs_lengths: list type
        - max_len(targer sentences max len in batch): T_y
        """
        if max_len is None: max_len = self.max_len
        
        inputs = self.start_token(hidden.size(0)) # (B, 1)
        embeded = self.embed(inputs) # (B, 1, m_d)
        if self.use_dropout:
            embeded = self.dropout(embeded)
            
        # prepare for whole targer sentence scores
        scores = []
        attn_weights = []

        for i in range(max_len):
            # context vector: previous hidden(s{i-1}), encoder_outputs(O_e) > context(c{i}), weights
            # - context: (B, 1, n_d)
            # - weights: (B, 1, T_x)
            context, weights = self.attention(hidden, enc_outputs, enc_outputs_lengths, 
                                              return_weight=self.return_weight)
            attn_weights.append(weights.squeeze(1))
            
            # concat context & embedding vectors: (B, 1, m_d+n_d)
            gru_input = torch.cat([embeded, context], 2)
            
            # gru((context&embedding), previous hidden)
            # output hidden(s{i}): (1, B, n_d)
            _, hidden = self.gru(gru_input, hidden.transpose(0, 1))
            hidden = hidden.transpose(0, 1)  # change shape to (B, 1, n_d) again
            
            # concat context and new hidden vectors: (B, 1, 2*n_d)
            concated = torch.cat([hidden, context], 2)
            
            # get score: (B, V_d)
            score = self.linear(concated.squeeze(1))
            scores.append(score)
            
            # greedy method
            decoded = self.decode_method(score, dec_method=self.dec_method, ktop=self.ktop)  # (B)
            embeded = self.embed(decoded).unsqueeze(1) # next input y{i-1} (B, 1, m_d)
            if self.use_dropout:
                embeded = self.dropout(embeded)

        # column-wise concat, reshape!! 
        # scores = [(B, V_d), (B, V_d), (B, V_d)...] > (B, V_d*max_len)
        # attn_weights = [(B, T_x), (B, T_x), (B, T_x)...] > (B*max_len, T_x)
        scores = torch.cat(scores, 1)
        return scores.view(inputs.size(0)*max_len, -1), torch.cat(attn_weights)

    def decode_method(self, score, dec_method='greedy', ktop=5):
        prob, decoded = score.max(1)
        if dec_method == 'greedy':
            return decoded
        elif dec_method == 'beam':
            pass



    def decode(self, hidden, enc_outputs, enc_outputs_lengths, eos_idx=3, max_len=50):
        
        inputs = self.start_token(hidden.size(0))  # (1, 1)
        embeded = self.embed(inputs)  # (1, 1, m_d)
        if self.use_dropout:
            embeded = self.dropout(embeded)
        
        decodes = [] 
        attn_weights = []
        decoded = torch.LongTensor([self.sos_idx]).view(1, -1)
        
        while (decoded.item() != eos_idx):
            # context: (1, 1, n_d)
            # weights: (1, 1, T_x)
            context, weights = self.attention(hidden, enc_outputs, enc_outputs_lengths, 
                                              return_weight=self.return_weight)
            attn_weights.append(weights.squeeze(1))  # (1, T_x)
            gru_input = torch.cat([embeded, context], 2)  # (1, 1, m_d+n_d)
            _, hidden = self.gru(gru_input, hidden.transpose(0, 1))  # (1, 1, n_d)
            hidden = hidden.transpose(0, 1)
            concated = torch.cat([hidden, context], 2)  # (1, 1, 2*n_d)
            score = self.linear(concated.squeeze(1))  # (1, 2*n_d) -> # (1, V_d)
            decoded = score.max(1)[1]  # (1)
            decodes.append(decoded)
            embeded = self.embed(decoded).unsqueeze(1) # (1, 1, m_d)
            if self.use_dropout:
                embeded = self.dropout(embeded)
            
            if len(decodes) >= max_len:
                break
        
        return torch.cat(decodes), torch.cat(attn_weights)