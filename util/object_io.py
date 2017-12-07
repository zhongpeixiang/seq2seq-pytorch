"""
This is where all methods for interacting with the disk reside
save_object: save an object to disk as a pickle file
"""

import pickle

# Save an object to disk as a pickle file
def save_object(obj, filename):
    print("Saving to " + filename)
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# Load an object from disk
def load_object(filename):
    print("Loading from " + filename)
    with open(filename, 'rb') as input:
        return pickle.load(input)

