import os
from itertools import zip_longest


def return_training():
	training_data = []
	#Reading in Data from the Train Set
	with open("train_small.txt") as textfile1, open("train_small_output_gold.txt") as textfile2: 
	    for x, y in zip_longest(textfile1, textfile2):
	    	characters = list(x.strip('\n'))
	    	characters_gold = list(y.strip('\n'))
	    	tuple_char = (characters, characters_gold)
	    	training_data.append(tuple_char)

	return training_data

if __name__ == '__main__':
	training_data_reported = return_training()
	print(training_data_reported)

