from random import shuffle

EXTERNAL_DATA_DIR = "/media/external/peixiang/ACL2018/data/"
INTERNAL_DATA_DIR = "./data/"

def train_val_test_split(input_filename, val_ratio = 0.2, test_ratio = 0.1):
    train_filename = input_filename.replace(".txt", "_train.txt")
    val_filename = input_filename.replace(".txt", "_val.txt")
    test_filename = input_filename.replace(".txt", "_test.txt")

    with open(input_filename) as input_file:
        lines = input_file.read().splitlines()
    
    # Shuffle lines
    shuffle(lines)

    lines_len = len(lines)
    num_train_lines = int((1 - val_ratio - test_ratio) * lines_len)
    num_val_lines = int(val_ratio * lines_len)

    with open(train_filename, "w") as f:
        print("Writing into training file...")
        for line in lines[:num_train_lines]:
            f.write(line)
    
    with open(val_filename, "w") as f:
        print("Writing into validation file...")
        for line in lines[num_train_lines: num_train_lines + num_val_lines]:
            f.write(line)

    with open(test_filename, "w") as f:
        print("Writing into testing file...")
        for line in lines[num_train_lines + num_val_lines:]:
            f.write(line)

train_val_test_split(EXTERNAL_DATA_DIR + "opensub/OpenSubData/s_given_t_dialogue_length2_3_result.txt")
    
