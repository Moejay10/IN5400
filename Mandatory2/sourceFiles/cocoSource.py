from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network

        Args:
            config: Dictionary holding neural network configuration

        Returns (creates):
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[number_of_cnn_features, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size        = config['vocabulary_size']
        self.embedding_size         = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes     = config['hidden_state_sizes']
        self.num_rnn_layers         = config['num_rnn_layers']
        self.cell_type              = config['cellType']

        # TODO: Task 1e
        
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.inputLayer = nn.Linear(self.number_of_cnn_features, self.hidden_state_sizes)

        self.rnn = RNN(self.embedding_size, self.hidden_state_sizes, self.num_rnn_layers, cell_type=self.cell_type)

        self.outputLayer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size)

        return

    def forward(self, cnn_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            cnn_features        : Features from the CNN network, shape[batch_size, number_of_cnn_features]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # TODO: Task 1e
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.
        # use self.rnn to calculate "logits" and "current_hidden_state"


        if current_hidden_state is None:
            input_layer = torch.tanh(self.inputLayer(cnn_features))
            initial_hidden_state = input_layer.repeat(repeats=(self.num_rnn_layers, 1, 1))
        else:
            initial_hidden_state = current_hidden_state

        logits, current_hidden_state_out = self.rnn(xTokens, initial_hidden_state, self.outputLayer, self.Embedding, is_train)

        return logits, current_hidden_state_out

######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers)
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells

        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type

        # TODO: Task 1d
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type".

        if self.cell_type == "RNN":
            cell = [RNNCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cell.extend([RNNCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])  # Calculate current state by iterating through each cell

            self.cells = nn.ModuleList(cell)

        else:
            self.cell_type == "GRU"
            cell = [GRUCell(hidden_state_size, input_size)]
            new_input_size = hidden_state_size
            cell.extend([GRUCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])  # Calculate current state by iterating through each cell

            self.cells = nn.ModuleList(cell)

        return


    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] # Truncated_backprop_length
        else:
            seqLen = 40 # Max sequence length to be generated

        # ToDo
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.

        current_states = list(torch.unbind(initial_hidden_state)) # Empty tensor to store the hidden states for all layers in a recurrence
        num_rnn_layers = list(initial_hidden_state.size())[0]
        batch_size = list(xTokens.size())[0]

        input_embedding_vectors = Embedding(xTokens) # Getting input embedding vectors

        if is_train == True:
            list_logit = []
            for i in range(seqLen):                     # Iterates through all 30 potential words
                state_old = input_embedding_vectors[:, i, :]  # Get the embedding vector for the correct input word

                for j in range(num_rnn_layers):         # Calculate current state by iterating through each cell
                    state_new = self.cells[j](state_old, current_states[j]) # Forward the cell
                    current_states[j] = state_new       # Store the previous current states of each layer
                    state_old= state_new                # Update the previous cell

                logit = outputLayer(state_old)
                list_logit.append(logit)           # Forward the final state of the recurrent layer time instance
                                                    # into a fully connected (dense) layer to get values that each can be
                                                    # interpreted as a "logistic unit" (logit) for prediction.


            logits = torch.stack(list_logit, dim=1)         # Convert the logit values into tensor of size [batch_size, seqLen, vocabulary_size]
                                                            # containing the logit behind the predicted word for the entire reccurence

            current_state = torch.stack(current_states)     # Convert to the current_states of each layer into a tensor


        else:                                               # Not training
            list_logit = []
            state_old = input_embedding_vectors[:, 0, :]    # The first recurrence input is the word embedding for the first word
            for i in range(seqLen):
                for j in range(num_rnn_layers):
                    state_new = self.cells[j](state_old, current_states[j])
                    current_states[j] = state_new
                    state_old = state_new

                logit = outputLayer(state_old)
                list_logit.append(logit)

                pred = torch.argmax(logit, dim=1)                       # Predict the next word by locating the index of the word with the largest logit
                state_old = Embedding(pred)                             # Each later input is an embedding of a  predicted word.

            logits = torch.stack(list_logit, dim=1)
            current_state = torch.stack(current_states)

        return logits, current_state

########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_sizes = hidden_state_size
        self.input_size = input_size

        m = self.hidden_state_sizes + self.input_size
        h = self.hidden_state_sizes

        # TODO: Task 1b

        self.weight_u = torch.nn.Parameter(torch.randn(m, h)/np.sqrt(m))
        self.bias_u   = nn.Parameter(torch.zeros(1, h))

        self.weight_r = torch.nn.Parameter(torch.randn(m, h)/np.sqrt(m))
        self.bias_r   = nn.Parameter(torch.zeros(1, h))

        self.weight = torch.nn.Parameter(torch.randn(m, h)/np.sqrt(m))
        self.bias   = nn.Parameter(torch.zeros(1, h))


        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO: Task 1b

        temp = torch.cat((x, state_old), dim=1)

        update_gate = torch.sigmoid(torch.mm(temp, self.weight_u) + self.bias_u)

        reset_gate = torch.sigmoid(torch.mm(temp, self.weight_r) + self.bias_r)

        temp2 = torch.cat((x, reset_gate*state_old), dim=1)
        candidate_cell = torch.tanh(torch.mm(temp2, self.weight) + self.bias)

        # Final cell
        state_new = update_gate*state_old + (1 - update_gate)*candidate_cell

        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size
        self.input_size = input_size

        m = self.hidden_state_size + self.input_size
        h = self.hidden_state_size


         # TODO: Task 1a

        self.weight = torch.nn.Parameter(torch.randn(m, h)/np.sqrt(m))

        self.bias   = nn.Parameter(torch.zeros(1, h))

        return


    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO: Task 1a

        state_new = torch.tanh(torch.mm(torch.cat((x, state_old), dim=1), self.weight) + self.bias)

        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """

    # TODO: Task 1c

    eps = 0.0000000001                                        # Used to not divide on zero
    logit = torch.transpose(logits, 1, 2)                     # Transforms into correct dimension
    loss = F.cross_entropy(logit, yTokens, reduction="none")  # Finds the correct dimension and calculates the total loss
    losses = torch.masked_select(loss, yWeights.eq(1))        # Loss for non-empty words
    sumLoss  = torch.sum(losses)                              # Sum of all non-empty words
    meanLoss = torch.mean(losses)                             # Mean of all non-empty words

    return sumLoss, meanLoss
