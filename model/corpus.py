"""
This is the Corpus class for indexing and filtering words in the corpus
It has three dictionaries: word2index, index2word and word2count
It has two methods for filtering words based on min word count and vocabulary size limit
"""
PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

class Corpus:
    def __init__(self, name):
        """
        name: name of the corpus
        """
        self.name = name
        self.trimmed = False
        self.filtered = False
        self.init_dict()
        self.word2count_copy = None

    def init_dict(self, keep_words=None):
        # initialize dictionaries
        self.word2index = {"PAD": 0, "UNK": 1, "SOS": 2, "EOS": 3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "UNK", 2: "SOS", 3: "EOS"}
        self.n_words = 4 # Count default tokens

        if keep_words:
            for word in keep_words:
                self.index_word(word)
    
    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)
        self.word2count_copy = self.word2count.copy()

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count_copy.items():
            if v >= min_count:
                keep_words.append(k)

        print('Trim: keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.init_dict(keep_words)

    # Keep a fixed vocab size and remove the rest based on word counts
    def filter_vocab(self, vocab_size):
        if self.filtered or len(self.word2count) <= vocab_size: return
        self.filtered = True

        keep_words = []

        sorted_word_count = sorted(self.word2count_copy.items(), key=lambda m: m[1], reverse=True)
        
        keep_words = [k for k, v in sorted_word_count[:vocab_size]]

        print('Filter: keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.init_dict(keep_words)

    
    

