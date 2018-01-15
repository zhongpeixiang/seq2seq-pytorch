"""
All configuration parameters are here
"""

####################
### Hyper-parameters
####################
# GPU and IO
USE_CUDA = True # If True, use GPU
GPU_ID = 1 # GPU device id
SAVE_CORPUS = False # Save preprocessed corpus data (corpus, pairs) into pickle file for easy reload
LOAD_CORPUS = True # Load preprocessed corpus, pairs data
SAVE_MODEL = False # Save PyTorch encoder, decoder model
LOAD_MODEL = False # Load PyTorch encoder, decoder model

# Filter words
CREATE_CORPUS_ONLY = False # If True, the seq2seq script creates preprocessed corpus files only
EXTERNAL_DATA_DIR = "/media/external/peixiang/ACL2018/data/" # Directory for data in external hard disk
INTERNAL_DATA_DIR = "./data/" # Directory for data in internal hard disk
USE_DIR = EXTERNAL_DATA_DIR
DATA_FILE = "opensub/OpenSubData/s_given_t_dialogue_length2_3_result.txt" # File path for raw text file
CORPUS = "opensub" # Name of the corpus used
MIN_COUNT = 10 # Min frequency of filtered words
VOCAB_SIZE = 10000 # Max vocab size of filtered words
LIMIT_PAIRS = 10000000 # Limit the number of pairs extracted from the data
REPLACE_UNK = False # If True, replace unknown words by UNK, if False, remove pairs that contain unknown words
MIN_LENGTH = 3 # Min number of words in a sentence for filtering purpose
MAX_LENGTH = 10 # Max number of words in a sentence for filtering purpose
REVERSE_INPUT = True # Reverse the words in input sentence
SEED = 412 # Seed number for train, val and test sets split
LOAD_WORD2VEC = True # If True, use pre-trained word2vec word embeddings
WORD2VEC_PATH = "./data/word2vec/GoogleNews-vectors-negative300.bin" # Path for pre-trained word2vec word embeddings
SAVE_NUMPY_WORD2VEC = False # If True, save word2vec embedding to numpy matrix
LOAD_NUMPY_WORD2VEC = True # If True, load word2vec embedding from numpy matrix
USE_AFFECT_EMBEDDING = True # If True, use affect wrod embedding
AFFECT_EMBEDDING_PATH = "./data/vad_words/word2vad.pkl" # Path of affect embedding dictionary for 13,915 words
AFFECT_EMBEDDING_STRENGTH = 0.01 # Weights for word2vec and affect embedding, usually in [0.01, 0.05]
USE_AFFECT_ATTN = False # If True, use affective attention mechanism
AFFECT_LOSS_STRENGTH = 0.1 # Weight for affective loss between output sentence and target sentence


# Model configuration
attn_model = 'dot' # Attentional model type: dot, general and concat
embedding_size = 300 # Embedding layer size
hidden_size = 1024 # Hidden layer size
n_layers = 2 # Number of GRU layers
dropout = 0.1 # Dropout strength
batch_size = 64 # Batch size
ordered_batch = False # If True, sort training samples by input sequence length and get batch that contains samples of equal input sequence length
val_ratio = 0.2 # Ratio of validation set to total data set
test_ratio = 0.1 # Ratio of test set to total data set
model_identifier = "eval"

# Training configuration
clip = 5 # Gradient clipping to avoid gradient exploding problem
teacher_forcing_ratio = 1 # (0, 1), controls the frequency of using target word for training instead of using predicted word for training in decoding process
learning_rate = 0.0001 # Initail learning rate
decoder_learning_ratio = 5 # Learning rate ratio for decoder
beam_size = 4 # Beam size for decoding process
alpha = 0.6 # Length penalty for beam search
n_epochs = 80000 # Number of training cycles, one cycle trains one batch of samples
epoch = 0 # Starting epoch at 0
early_stopping = False # If True, the training stops when consecutive validation errors are increasing
print_every = 200 # Print training summary every a few epochs
n_validations = 40 # Number of validation batches per validation
evaluate_every = 200 # Evaluate model using evaluation dataset every a few epcohs
save_every = 2000 # Save encoder and decoder every a few epcohs