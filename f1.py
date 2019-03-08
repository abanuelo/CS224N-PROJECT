import os
from itertools import zip_longest

def compute_F1_scores():
    F1_micro = 0
    F1_macro = 0
    F1_arr = []
    TP_macro = 0
    FP_macro = 0 
    FN_macro = 0

    with open("./data/output_deepcut_bin.txt") as tgt_file, open("./data/test_output_gold.txt") as gold_file:
        for tgt, gold in zip_longest(tgt_file, gold_file):
            characters = list(tgt.strip('\n'))
            characters_gold = list(gold.strip('\n'))
            TP= 0
            FP= 0
            FN = 0
            for i in range(len(characters)):
                if characters[i] == characters_gold[i] and characters[i] == '1':
                    TP += 1
                    TP_macro += 1
                elif characters[i] != characters_gold[i] and characters[i] == '0':
                    FP += 1 
                    FP_macro += 1
                elif characters[i] != characters_gold[i] and characters[i] == '1':
                    FN += 1
                    FN_macro += 1
            if (TP + FP > 0 and TP + FN > 0):
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                if (precision + recall > 0):
                    F1 = (2*precision*recall) / (precision+recall)
                else:
                    F1 = 0
            else:
                F1 = 0
            F1_arr.append(F1)
    F1_micro = sum(F1_arr) / len(F1_arr)
    precision_macro = TP_macro / (TP_macro + FP_macro)
    recall_macro = TP_macro / (TP_macro + FN_macro)
    F1_macro = (2*precision_macro*recall_macro) / (precision_macro+recall_macro)
    print("F1 Micro Score", F1_micro)
    print("F1 Macro Score", F1_macro)
    return F1_micro, F1_macro

def main():
    compute_F1_scores()

if __name__ == '__main__':
    main()