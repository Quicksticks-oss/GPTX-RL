import numpy as np
import argparse
import json

def main(input_file, output_file, full):
    if input_file == None or output_file == None:
        print('Please pass an argument for input and output.')
        return
    with open('train.txt', 'r', encoding='utf-8') as f:
        train_data = repr(f.read())

    train_data = list(train_data)
    train_data.pop(0)
    train_data.pop(len(train_data)-1)
    train_data = ''.join(train_data)

    tokens = np.array(train_data.split())
    tokens.sort()
    if full == False:
        tokens = np.unique(tokens)
    
    print('Vocab Size:', len(tokens))
    
    json_list = {'vocab': ['']}

    for _ in range(len(tokens)):
        json_list['vocab'].append(tokens[_])
    
    with open(output_file, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(json_list))
    
    print('Completed!')

if __name__ == '__main__':
    print('State-RL Vocab Generator v1.0')
    parser = argparse.ArgumentParser(description="GTPX-RL Vocab Generator v1.0")
    parser.add_argument("-i", "--input", default=None, help = "Input txt file")
    parser.add_argument("-o", "--output", default='vocab.json', help = "Output vocab file")
    parser.add_argument("-full", "--full", default=False, help = "(true or false) this will return the full vocab without removal of dupes. (might break)")
    args = parser.parse_args()
    main(args.input, args.output, args.full)
