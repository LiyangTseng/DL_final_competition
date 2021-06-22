import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

MAX_SEQ_LEN = 1007
class LSTM(nn.Module):
    def __init__(self, hidden_size, category_num, bidirectional, device, dropout=0):
        super(LSTM, self).__init__()
        ''' 
        LSTM input:  (batch, 20, MAX_SEQ_LEN)
        LSTM output: (batch, 20, hidden_size * (2 if bidirectional else 1))
        '''
        self.device = device
        self.LSTM = nn.LSTM(MAX_SEQ_LEN, hidden_size, dropout=dropout, batch_first=True, bidirectional=True)
        if bidirectional:
            lstm_out_size = hidden_size*2
        else:
            lstm_out_size = hidden_size

        self.Linear = nn.Linear(lstm_out_size*MAX_SEQ_LEN, category_num)
        self.Activation = nn.Tanh()

    def forward(self, packed_input):
        # packed_input: (batch_sum_seq_len, 300)
        packed_output, _ = self.LSTM(packed_input)    
        # packed_output: (batch_sum_seq_len, 10*2)
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        shape = unpacked_output.size(0), MAX_SEQ_LEN-unpacked_output.size(1), unpacked_output.size(2) # pad seq_len to MAX_LEN
        unpacked_output = torch.cat((unpacked_output, torch.zeros(shape, device=self.device)), dim=1)
        # unpacked_output: (batch, max_seq_len, hidden_size*num_direction)
        flatten = unpacked_output.view(unpacked_output.shape[0], -1)   
        # flatten: (batch, max_seq_len * hidden_size * num_direction)
        out = self.Linear(flatten)              
        out = self.Activation(out)

        return out
