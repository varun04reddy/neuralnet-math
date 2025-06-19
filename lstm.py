import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # defining the weights for the gates

        self.W_input = nn.Linear(input_dim, hidden_dim)
        self.U_input = nn.Linear(hidden_dim, hidden_dim)

        self.W_forget = nn.Linear(input_dim, hidden_dim)
        self.U_forget = nn.Linear(hidden_dim, hidden_dim)

        self.W_cell = nn.Linear(input_dim, hidden_dim)
        self.U_cell = nn.Linear(hidden_dim, hidden_dim)

        self.W_output = nn.Linear(input_dim, hidden_dim)
        self.U_output = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, x, h_prev, c_prev):
        # handles the gating computations at a single time step

        # input gate (how much to update the cell with new information)
        i = torch.sigmoid(self.W_input(x) + self.U_input(h_prev))
        # forget gate (how much of the old cell to keep)
        f = torch.sigmoid(self.W_forget(x) + self.U_forget(h_prev))
        # output gate (How much of the cell state do we expose to the next layer output)
        o = torch.sigmoid(self.W_output(x) + self.U_output(h_prev))

        # cell candidate (what we want to add to the cell)
        c_hat = torch.tanh(self.W_cell(x) + self.U_cell(h_prev))

        # update the cell
        c = f * c_prev + i * c_hat
        # update the hidden state
        h = o * torch.tanh(c)
        return h, c
    

class MathLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = LSTMCell(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # handles the forward pass for the entire sequence of time steps

        batch_size, seq_len = x.size()
        embedded = self.embedding(x)

        h = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_dim, device=x.device)

        # iterate through each time step
        for t in range(seq_len):
            h, c = self.lstm_cell(embedded[:, t, :], h, c)

        output = self.fc(h)
        return output.squeeze(1)
    
    def get_weights(self):

        return {
            "W_i_weight": self.lstm_cell.W_input.weight.data,
            "W_i_bias": self.lstm_cell.W_input.bias.data,
            "U_i_weight": self.lstm_cell.U_input.weight.data,
            "U_i_bias": self.lstm_cell.U_input.bias.data,

            "W_f_weight": self.lstm_cell.W_forget.weight.data,
            "W_f_bias": self.lstm_cell.W_forget.bias.data,
            "U_f_weight": self.lstm_cell.U_forget.weight.data,
            "U_f_bias": self.lstm_cell.U_forget.bias.data,

            "W_o_weight": self.lstm_cell.W_output.weight.data,
            "W_o_bias": self.lstm_cell.W_output.bias.data,
            "U_o_weight": self.lstm_cell.U_output.weight.data,
            "U_o_bias": self.lstm_cell.U_output.bias.data,

            "W_c_weight": self.lstm_cell.W_cell.weight.data,
            "W_c_bias": self.lstm_cell.W_cell.bias.data,
            "U_c_weight": self.lstm_cell.U_cell.weight.data,
            "U_c_bias": self.lstm_cell.U_cell.bias.data,
        }

