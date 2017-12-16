def transform_opensubtitles(input_filename, output_filename, dict_filename):
    with open(input_filename) as input_file:
        lines = input_file.read().splitlines()
    lines_len = len(lines)
    print("Processing {0} lines...".format(lines_len))
    with open(dict_filename) as dict_file:
        dictionary = dict_file.read().splitlines()
    with open(output_filename, "w") as output_file:
        count = 0
        for line in lines:
            count += 1
            Q = line.split("|")[0]
            A = line.split("|")[1]

            Q_sent = " ".join([dictionary[int(idx) - 1] for idx in Q.split(" ")])
            A_sent = " ".join([dictionary[int(idx) - 1] for idx in A.split(" ")])

            sent = Q_sent.strip() + "\t" + A_sent.strip() + "\n"
            output_file.write(sent)
            
            if count % 10000 == 0:
                print("Processed {0} lines, progress: {1:.2f}%".format(count, 100*count/lines_len))

transform_opensubtitles("./data/opensub/OpenSubData/s_given_t_dialogue_length2_3.txt", "./data/opensub/OpenSubData/s_given_t_dialogue_length2_3_result.txt", "./data/opensub/OpenSubData/dictionary.txt")