"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        # 1) An embedding layer
        self.embedding = nn.Embedding(output_size, emb_size)

        # 2) A recurrent layer
        if model_type == "RNN":
            self.recurrent = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)
        elif model_type == "LSTM":
            self.recurrent = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)
        
        # 3) A single linear layer with a (log)softmax layer for output
        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
       
        # 4) A dropout layer
        self.dropout = nn.Dropout(dropout)

        # 5) Attention linear layer
        if attention:
            self.attention_linear = nn.Linear(encoder_hidden_size + emb_size, emb_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        if self.model_type == "LSTM":
            q = hidden[0]
        else:
            q = hidden
        
        q = q.squeeze(0)  # (N, hidden_dim)
        K = encoder_outputs  # (N, T, hidden_dim)
        q_expand = q.unsqueeze(1).expand_as(K) # (N, T, hidden_dim)
        
        # Compute cosine similarity
        scores = F.cosine_similarity(q_expand, K, dim=-1)  # (N, T)
    
        # Normalize attention probabilities
        attention_prob = F.softmax(scores, dim=-1).unsqueeze(1)  # (N, 1, T)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer and is fed into the linear layer    #
        #       you created in the init section. The output of this layer is fed    #
        #       as input vector to your recurrent layer. Refer to the diagram       #
        #       provided in the Jupyter notebook for further clarifications. note   #
        #       that attention is only applied to the hidden state of LSTM.         #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # 1) Apply the dropout to the embedding layer
        embedded = self.embedding(input) # (N, 1, emb_size)
        embedded = self.dropout(embedded)

        # 2) Attention mechanism
        if self.attention and encoder_outputs is not None:
            attention_prob = self.compute_attention(hidden, encoder_outputs)
            context = torch.bmm(attention_prob, encoder_outputs) # (N, 1, encoder_hidden_size)

            # Concatenate context vector with embedded input
            rnn_input = torch.cat((context, embedded), dim=2) 
            rnn_input = self.attention_linear(rnn_input) # (N, 1, emb_size)
        else:
            rnn_input = embedded

        # 3) Feed the input vector to the recurrent layer
        output, hidden = self.recurrent(rnn_input, hidden)

        # Apply linear layer and log-softmax activation to output tensor
        output = self.linear(output)
        output = self.log_softmax(output)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
