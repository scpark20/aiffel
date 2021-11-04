import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kspon_jamo import n_symbols, text_to_tokens, tokens_to_text, SOS, EOS

class Model(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm1d(hidden_channels),
                                  nn.ReLU())
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels//2, num_layers=3, 
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.out_layer = nn.Linear(hidden_channels, output_channels)
        
    def forward(self, inputs, input_lengths):
        # inputs : (batch, mel_length, channel)
        # input_lengths : (batch)
        
        '''Convolution'''
        # (batch, time, channel)
        x = self.conv(inputs.transpose(1, 2)).transpose(1, 2)
        
        '''LSTM'''
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        # (batch, time channel)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        '''Get Log-Probability'''
        # (batch, time, n_symbols)
        log_probs = F.log_softmax(self.out_layer(x), dim=2)
        
        return log_probs

# Remove blank, SOS, EOS tokens
def refine_tokens(tokens):
    prev_token = None
    new_tokens = []
    for token in tokens:
        if prev_token == token:
            continue
        
        if token == 0 or token == SOS or token == EOS:
            prev_token = token
            continue
            
        new_tokens.append(token)
        prev_token = token
        
    return np.array(new_tokens)