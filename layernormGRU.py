import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import numpy as np


class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, layernorm=False,
                 bidirectional=False, bias = True, use_cuda=False, return_all_hidden = False):
        super(LayerNormGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layernorm = layernorm
        self.batch_first = batch_first
        self.bidrectional = bidirectional
        self.return_all_hidden = return_all_hidden
        self.num_directions = 2 if self.bidrectional else 1
        self.bias = bias
        self.gate_num = 3
        self.use_cuda = use_cuda

    def forward(self, inputs, hidden=None):
        """
        input:
        * inputs: seq_len, B, input_size
        * hidden: num_layers * num_directions, B, hidden_size
        output:
        * output: seq_len, B, hidden_size * num_directions
        * hidden: num_layers * num_directions, B, hidden_size
        """
        is_packed = isinstance(inputs, PackedSequence)
        if is_packed:
            inputs, batch_sizes = inputs
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = inputs.size(0) if self.batch_first else inputs.size(1)

        if hidden is None:
            hidden = self.init_hidden(inputs, max_batch_size)

        func = StackedGRU(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=self.bidrectional,
                           layernorm=self.layernorm,
                           return_all_hidden=self.return_all_hidden,
                           batch_first=self.batch_first,
                           batch_sizes=batch_sizes,
                           is_packed=is_packed)

        output, hidden = func(inputs, hidden)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def init_hidden(self, inpt, max_batch_size):
        hx = inpt.new_zeros(self.num_layers * self.num_directions, max_batch_size, self.hidden_size,
                            requires_grad=False)
        if self.use_cuda:
            hx = hx.cuda()
        return hx


class StackedGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, layernorm=False,
                 return_all_hidden=False, batch_sizes=None, batch_first=False, is_packed=False):
        super(StackedGRU, self).__init__()
        # to do: add is_packed
        self.batch_first = batch_first
        self.layernorm = layernorm
        self.bidirec = bidirectional
        self.return_all_hidden = return_all_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if self.bidirec else 1
        self.num_layers = num_layers
        self.build_layers(input_size, hidden_size)
        # packed seq
        self.batch_sizes = batch_sizes
        self.is_packed = is_packed

    def build_layers(self, input_size, hidden_size):
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GRUCell(input_size, hidden_size, layernorm=self.layernorm))
            input_size = hidden_size
        if self.bidirec:
            input_size = self.input_size
            self.r_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.r_layers.append(GRUCell(input_size, hidden_size, layernorm=self.layernorm))
                input_size = hidden_size

    def forward(self, inputs, hidden, batch_sizes=None):
        """
        * input:
        inputs: 'tensor(T, B, D)' if packed, 'tensor(T*B, D)'
        hidden: 'tensor(num_layers * num_directions, B, H)'

        * return:
        output: 'tensor(num_layers, T, B, 2H)' if return_all_hiddens else last layer 'tensor(T, B, 2H)'
        hidden 'tensor(num_layers*num_directions, B, H)'
        """
        if self.bidirec:
            # output (num_layers, T, B, 2H)
            # last_hidden (num_layers*num_directions, B, H)
            # forward: idx of time t ~ (0, 1, ..., T-1)
            f_idx = [i for i in range(self.num_layers * self.num_directions) if i % 2 == 0]
            f_all_outputs, f_last_hidden = self._forward(self.layers, inputs, hidden[f_idx, :])

            # backward:
            r_inputs = self._flip(inputs, 0)  # (T, B, H) idx of time t ~ (T-1, ... , 0)
            b_idx = [i for i in range(self.num_layers * self.num_directions) if i % 2 != 0]
            b_all_outputs, b_last_hidden = self._forward(self.r_layers, r_inputs, hidden[b_idx, :])

            # concate layers
            # f: hidden[T-1], b: hidden[0]
            output = torch.cat([f_all_outputs, b_all_outputs], -1)
            idx = [int(i / self.num_directions) if i % 2 == 0 else \
                       i + int(((self.num_layers * self.num_directions) - i) / 2) \
                   for i in range(self.num_layers * self.num_directions)]
            hidden = torch.cat([f_last_hidden, b_last_hidden])[idx, :]

            if self.return_all_hidden:
                return output, hidden
            return output[-1], hidden

        else:
            f_all_outputs, f_last_hidden = self._forward(self.layers, inputs, hidden)
            if self.return_all_hidden:
                return f_all_outputs, f_last_hidden
            return f_all_outputs[-1], f_last_hidden

    def init_hidden(self, batch_size):
        # init_hidden
        hidden = torch.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size))
        return hidden

    def _forward(self, layers, inputs, hidden, batch_sizes=None):
        """
        * input:
        layers: nn.ModuleList for one direction layers
        inp: T, B, D
        hid: num_layers, B, H (init hidden)

        * return:
        all_outputs: all layers a forward or backward layer
        tensor(num_layers, T, B, H)
        last_hidden:
        tensor(num_layers, B, H)
        """
        # todo: add is_packed
        assert isinstance(layers, nn.ModuleList)
        inp = inputs

        all_outputs = []
        for l_idx, layer in enumerate(layers):
            hid = hidden.chunk(3, 0)[l_idx].squeeze(0)  # init hidden: 1, B, H --> B, H
            output_ith_layer = []
            for t in range(inp.size(0)):
                hid = layer(inp[t], hid)
                output_ith_layer.append(hid)
            output_ith_layer = torch.stack(output_ith_layer)  # T, B, H
            inp = output_ith_layer
            all_outputs.append(output_ith_layer)

        last_hidden = torch.stack([out[-1] for out in all_outputs])  # num_layer, B, H
        return torch.stack(all_outputs), last_hidden

    def _flip(self, x, dim):
        """
        https://discuss.pytorch.org/t/optimizing-diagonal-stripe-code/17777/16
        """
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, layernorm=False, gate_num=3):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.bias = bias
        self.hidden_size = hidden_size
        self.layernorm = layernorm
        self.gate_num = gate_num

        self.weight_ih = nn.Linear(input_size, gate_num * hidden_size, bias=bias)
        self.weight_hh = nn.Linear(hidden_size, gate_num * hidden_size, bias=bias)

        self.lm_r = nn.LayerNorm(hidden_size)
        self.lm_i = nn.LayerNorm(hidden_size)
        self.lm_n = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden):
        """
        inputs:
        * inputs: B, input_size
        * hidden: B, hidden_size
        output:
        * hy: B, hidden_size
        """
        gi = self.weight_ih(inputs)
        gh = self.weight_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        a_r = i_r + h_r
        a_i = i_i + h_i
        if self.layernorm:
            a_r = self.lm_r(a_r)
            a_i = self.lm_i(a_i)

        resetgate = F.sigmoid(a_r)
        inputgate = F.sigmoid(a_i)

        a_n = i_n + resetgate * h_n
        if self.layernorm:
            a_n = self.lm_n(a_n)

        newgate = F.tanh(a_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy