from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input modulation gate weights and bias
        self.Wgx = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.Wgh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bg = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Input gate weights and bias
        self.Wix = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.Wih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bi = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Forget gate weights and bias
        self.Wfx = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.Wfh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bf = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Output gate weights and bias
        self.Wox = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.Woh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bo = nn.Parameter(torch.Tensor(hidden_dim))
        
        # Output layer weights and bias (for prediction p(t))
        self.Wph = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.bp = nn.Parameter(torch.Tensor(output_dim))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Xavier/Glorot initialization for weights
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)
            elif 'b' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1 for better gradient flow
                if 'bf' in name:
                    nn.init.ones_(param)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        device = x.device
        
        # Initialize hidden state and cell state to zeros
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Process sequence through time
        for t in range(self.seq_length):
            x_t = x[:, t, :]  # (batch_size, input_dim)
            
            # Input modulation gate: g(t) = tanh(Wgx * x(t) + Wgh * h(t-1) + bg)
            g = torch.tanh(x_t @ self.Wgx.t() + h @ self.Wgh.t() + self.bg)
            
            # Input gate: i(t) = sigmoid(Wix * x(t) + Wih * h(t-1) + bi)
            i = torch.sigmoid(x_t @ self.Wix.t() + h @ self.Wih.t() + self.bi)
            
            # Forget gate: f(t) = sigmoid(Wfx * x(t) + Wfh * h(t-1) + bf)
            f = torch.sigmoid(x_t @ self.Wfx.t() + h @ self.Wfh.t() + self.bf)
            
            # Output gate: o(t) = sigmoid(Wox * x(t) + Woh * h(t-1) + bo)
            o = torch.sigmoid(x_t @ self.Wox.t() + h @ self.Woh.t() + self.bo)
            
            # Cell state: c(t) = g(t) * i(t) + c(t-1) * f(t)
            c = g * i + c * f
            
            # Hidden state: h(t) = tanh(c(t)) * o(t)
            h = torch.tanh(c) * o
        
        # Final output: p(t) = Wph * h(t) + bp
        p = h @ self.Wph.t() + self.bp
        
        # Return logits (softmax will be applied by CrossEntropyLoss)
        return p