import os
from itertools import zip_longest as zip

#Reconverts Binary 0-1 Data Back to its Thai Equivalent for Qualitative Analysis
def reconvert(inp, tgt, output_file):
    with open(inp) as tgt_file, open(tgt) as gold_file:
        for tgt, gold in zip(tgt_file, gold_file):
            characters = list(tgt.strip('\n'))
            characters_gold = list(gold.strip('\n'))
            #iterate and output text to a different file
            lineToInsert = ""
            for i in range(len(characters_gold)):
                if characters_gold[i] == '0':
                    lineToInsert += characters[i]
                elif characters_gold[i] == '1':
                    lineToInsert = lineToInsert + characters[i] + "|"
                else:
                    raise Exception("Invalid Non-binary character detected!", characters_gold[i])
            lineToInsert += '\n'
            output_file.write(lineToInsert)

#Main method that consolidates all information used for reconverting model
def main():
    output_file = open('output_file.txt', 'a')
    inp = "./data/train_small_1.txt"
    tgt = "./outputs/test_local_outputs.txt"
    reconvert(inp, tgt, output_file)

if __name__ == '__main__':
    main()
