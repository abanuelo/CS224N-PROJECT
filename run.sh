#!/bin/bash

if [ "$1" = "train" ]; then
	touch model.bin
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-input=./data/train.txt --train-gold=./data/train_output_gold.txt --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt --cuda
elif [ "$1" = "test" ]; then
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py test --save-to=./models/model.bin --test-input=./data/test.txt --test-gold=./data/test_output_gold.txt --cuda
elif [ "$1" = "train_local" ]; then
	touch model.bin
	python run.py train --train-input=./data/train_small.txt --train-gold=./data/train_small_output_gold.txt \
        --batch-size=3 --valid-niter=100 --max-epoch=100 --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt --lr=0.001
elif [ "$1" = "test_local" ]; then
	mkdir -p outputs
    touch outputs/test_local_outputs.txt
    python run.py test model.bin ./data/dev_small.txt ./data/dev_small_output_gold.txt outputs/test_outputs_local.txt 
else
	echo "Invalid Option Selected"
fi
