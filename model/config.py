"""
All configuration parameters are here
"""

####################
### Hyper-parameters
####################
# GPU and IO
USE_CUDA = True
SAVE_OBJECT = False
LOAD_OBJECT = True

# Filter words
MIN_COUNT = 5
VOCAB_SIZE = 10000
MIN_LENGTH = 2 
MAX_LENGTH = 25

# Model configuration
attn_model = 'dot'
embedding_size = 500
hidden_size = 500
n_layers = 2
dropout = 0.1
batch_size = 128
val_ratio = 0.2
test_ratio = 0.1

# Training configuration
clip = 10
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5
n_epochs = 10000
epoch = 0
print_every = 100
evaluate_every = 200
save_every = 500