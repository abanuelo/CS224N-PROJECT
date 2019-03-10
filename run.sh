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
        --batch-size=5 --valid-niter=100 --max-epoch=60 --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt
elif [ "$1" = "test_local" ]; then
	mkdir -p outputs
	#rm -f outputs/test_local_outputs.txt
    #touch outputs/test_local_outputs.txt
    python run.py test --save-to=./models/model.bin --test-input=./data/dev_small.txt --test-gold=./data/dev_small_output_gold.txt --model-output=./outputs/test_local_outputs.txt 
else
	echo "Invalid Option Selected"
fi
