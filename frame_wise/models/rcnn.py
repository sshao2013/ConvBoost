import torch
import torch.nn as nn
from torch.autograd import Variable

# Code implementation from:  tinyurl.com/ybs4ndlv (On Attention Models for Human Activity Recognition)
# Author: Vishvak S Murahari


class RCNN(nn.Module):
    def __init__(self, input_dim, n_classes, DEVICE, num_layers=1, is_bidirectional=False, dropout=0.5,
                 attention_dropout=0.5):
        super(RCNN, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = 128
        hidden_dim = 128 * self.num_directions
        self.dropout_val = dropout
        self.attention_dropout_val = attention_dropout
        FILTER_SIZE = 5
        FILTER_STRIDE = 1
        NUM_FILTERS = 64
        self.conv2DLayer1 = nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1), )
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), )
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), )
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1),)
        self.relu4 = nn.ReLU()
        self.lstm = nn.LSTM(NUM_FILTERS * input_dim, self.hidden_size, num_layers, bidirectional=is_bidirectional,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense_layer = nn.Linear(hidden_dim, n_classes)
        self.num_layers = num_layers
        # parameters for attention
        # self.W_Y_t = nn.Parameter(torch.mul(torch.randn(hidden_dim, hidden_dim),0.01))
        # self.W_h = nn.Parameter(torch.mul(torch.randn(hidden_dim, hidden_dim),0.01))
        # self.softmax_tranform = nn.Parameter(torch.mul(torch.randn(1, hidden_dim),0.01))
        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim, 1)
        self.softmax_attention = torch.nn.Softmax(dim=0)
        self.DEVICE = DEVICE

    def forward(self, input, vis_attention=False):
        input = input.unsqueeze(1)
        convout1 = self.conv2DLayer1(input)
        convout1 = self.relu1(convout1)
        convout2 = self.conv2DLayer2(convout1)
        convout2 = self.relu2(convout2)
        convout3 = self.conv2DLayer3(convout2)
        convout3 = self.relu3(convout3)
        convout4 = self.conv2DLayer4(convout3)
        convout4 = self.relu4(convout4)
        # reshape to put them in the lstm
        lstm_input = convout4.permute(2, 0, 1, 3)
        lstm_input = lstm_input.contiguous()
        lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1], -1)
        # put things in lstm
        lstm_input = self.dropout(lstm_input)
        output, hidden = self.lstm(lstm_input, self.initHidden())
        # attention stuff
        past_context = output[:-1]
        current = output[-1]
        # logits = self.dense_layer(current)
        # return logits
        attention_layer1_output = self.attentionLayer1(past_context)
        attention_layer1_output = attention_layer1_output + current
        # attention_layer1_output = attention_layer1_output

        attention_layer1_output = self.tanh1(attention_layer1_output)
        attention_layer1_output = self.attention_dropout(attention_layer1_output)
        attention_layer2_output = self.attentionLayer2(attention_layer1_output)
        attention_layer2_output = attention_layer2_output.squeeze(2)
        # find weights
        attn_weights = self.softmax_attention(attention_layer2_output)
        # the cols represent the weights
        attn_weights = attn_weights.unsqueeze(2)
        new_context_vector = torch.sum(attn_weights * past_context, 0)
        # use this new context vector for prediction
        # add a skip connection
        new_context_vector = new_context_vector + current
        logits = self.dense_layer(new_context_vector)
        if vis_attention:
            return logits, attn_weights
        return logits

    def initHidden(self):
        h0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, 256, self.hidden_size), 0.08)).to(self.DEVICE)
        c0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, 256, self.hidden_size), 0.08)).to(self.DEVICE)
        return (h0, c0)
