import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.cuda.current_device()

class Attention(nn.Module):
    def __init__(self, hidden_size, hidden_size2=None, method='general'):
        super(Attention, self).__init__()
        """
        hidden_size: set hidden size same as decoder hidden size which is n_d (= 2*n_e)
        hidden_size2: only for concat method, if none then is same as hidden_size (n_d)
        (in paper notation is n', https://arxiv.org/abs/1409.0473)
        methods:
        - 'dot': dot product between hidden and encoder_outputs
        - 'general': encoder_outputs through a linear layer 
        - 'concat': concat (hidden, encoder_outputs)
        - 'paper': concat + tanh
        """
        self.method = method
        self.hidden_size = hidden_size 
        self.hidden_size2 = hidden_size2 if hidden_size2 else hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size) 
            # linear_weight shape: (out_f, in_f)
            # linear input: (B, *, in_f)
            # linear output: (B, *, out_f)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size*2, self.hidden_size2)
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size2))
        
        
    def forward(self, hidden, encoder_outputs, encoder_lengths=None, return_weight=False):
        """
        input:
        - hidden, previous hidden(= H): B, 1, n_d 
        - encoder_outputs, source context(= O): B, T_x, n_d
        - encoder_lengths: real lengths of encoder outputs
        - return_weight = return weights(alphas)
        output:
        - attentioned_hidden(= z): B, 1, n_d
        - weights(= w): B, 1, T_x
        """
        H, O = hidden, encoder_outputs
        # Batch(B), Seq_length(T), dimemsion(n)
        B_H, T_H, n_H = H.size()
        B_O, T_O, n_O = O.size()
        
        if B_H != B_O:
            msg = "Batch size is not correct, H: {} O: {}".format(H.size(), O.size())
            raise ValueError(msg)
        else:
            B = B_H
        
        # score: (B, 1, T_x)
        s = self.score(H, O) 
        
        # encoding masking
        if encoder_lengths is not None:
            mask = s.data.new(B, T_H, T_O) # (B, 1, T_x)
            mask = self.fill_context_mask(mask, sizes=encoder_lengths, v_mask=float('-inf'), v_unmask=0)
            s += mask
        
        # softmax: (B, 1, T_x)
        w = F.softmax(s, 2) 
        
        # attention: weight * encoder_hiddens, (B, 1, T_x) * (B, T_x, n_d) = (B, 1, n_d)
        z = w.bmm(c)
        if return_weight:
            return z, w
        return z
    
    def score(self, H, O):
        """
        inputs:
        - hiddden, previous hidden(= H): B, 1, n_d 
        - encoder_outputs, source context(= O): B, T_x, n_d
        """
        if self.method == 'dot':
            # bmm: (B, 1, n_d) * (B, n_d, T_x) = (B, 1, T_x)
            e = H.bmm(O.transpose(1, 2))
            return e
        
        elif self.method == 'general':
            # attn: (B, T_x, n_d) > (B, T_x, n_d)
            # bmm: (B, 1, n_d) * (B, n_d, T_x) = (B, 1, T_x)
            e = self.attn(O)
            e = H.bmm(e.transpose(1, 2))
            return e
        
        elif self.method == 'concat':
            # H repeat: (B, 1, n_d) > (B, T_x, n_d)
            # cat: (B, T_x, 2*n_d)
            # attn: (B, T_x, 2*n_d) > (B, T_x, n_d)
            # v repeat: (1, n_d) > (B, 1, n_d)
            # bmm: (B, 1, n_d) * (B, n_d, T_x) = (B, 1, T_x)
            cat = torch.cat((H.repeat(1, O.size(1), 1), O), 2)
            e = self.attn(cat)
            v = self.v.repeat(O.size(0), 1).unsqueeze(1)
            e = v.bmm(e.transpose(1, 2))
            return e
        
        elif self.method == 'paper':
            # add tanh after attention linear layer in 'concat' method
            cat = torch.cat((H.repeat(1, O.size(1), 1), O), 2)
            e = F.tanh(self.attn(cat))
            v = self.v.repeat(O.size(0), 1).unsqueeze(1)
            e = v.bmm(e.transpose(1, 2))
            return e
    
    def fill_context_mask(self, mask, sizes, v_mask, v_unmask):
        """Fill attention mask inplace for a variable length context.
        Args
        ----
        mask: Tensor of size (B, T, D)
            Tensor to fill with mask values. 
        sizes: list[int]
            List giving the size of the context for each item in
            the batch. Positions beyond each size will be masked.
        v_mask: float
            Value to use for masked positions.
        v_unmask: float
            Value to use for unmasked positions.
        Returns
        -------
        mask:
            Filled with values in {v_mask, v_unmask}
        """
        mask.fill_(v_unmask)
        n_context = mask.size(2)
        for i, size in enumerate(sizes):
            if size < n_context:
                mask[i,:,size:] = v_mask
        return mask