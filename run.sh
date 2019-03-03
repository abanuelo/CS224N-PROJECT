#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-input=./data/train.text --train-gold=./data/train_output_gold.txt --cuda
elif [ "$1" = "test" ]; then
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py test model.bin ./data/test.es ./data/test_output_gold.txt outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-input=./data/train_small.txt --train-gold=./data/train_small_output_gold.txt \
        --batch-size=2 --valid-niter=100 --max-epoch=101
elif [ "$1" = "test_local" ]; then
	mkdir -p outputs
    touch outputs/test_local_outputs.txt
    python run.py test model.bin ./data/dev_small.txt ./data/dev_small_output_gold.txt outputs/test_outputs_local.txt 
else
	echo "Invalid Option Selected"
fi
