"""
All configuration parameters are here
"""

####################
### Hyper-parameters
####################
# GPU and IO
USE_CUDA = True
GPU_ID = 0
SAVE_CORPUS = False
LOAD_CORPUS = True
SAVE_MODEL = False
LOAD_MODEL = True

# Filter words
CORPUS = "cornell"
MIN_COUNT = 5
VOCAB_SIZE = 10000
MIN_LENGTH = 2 
MAX_LENGTH = 25
REVERSE_INPUT = True

# Model configuration
attn_model = 'dot'
embedding_size = 500
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 64
val_ratio = 0.2
test_ratio = 0.1

# Training configuration
clip = 5
teacher_forcing_ratio = 1
learning_rate = 0.0001
decoder_learning_ratio = 5
beam_size = 4
n_epochs = 10000
epoch = 0
early_stopping = False
print_every = 100
evaluate_every = 1
save_every = 1000