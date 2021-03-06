#!/bin/bash

if [ "$1" = "train" ]; then
	touch model.bin
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-input=./data/train.txt --train-gold=./data/train_output_gold.txt \
	 --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt --cuda 
elif [ "$1" = "train_gru" ]; then
	touch model.bin
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-input=./data/train.txt --train-gold=./data/train_output_gold.txt \
	 --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt --cuda \
	 --model=gru_crf
elif [ "$1" = "test" ]; then
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py test --save-to=./models/model.bin --test-input=./data/test.txt --test-gold=./data/test_output_gold.txt --model-output=./outputs/test_local_outputs.txt --cuda
elif [ "$1" = "test_gru" ]; then
    touch outputs/test_outputs.txt
    CUDA_VISIBLE_DEVICES=0 python run.py test --save-to=./models/model.bin --test-input=./data/test.txt \
    --test-gold=./data/test_output_gold.txt --cuda --model=gru_crf
elif [ "$1" = "train_medium" ]; then
	touch ./models/model_medium.bin
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-input=./data/train_1000.txt --train-gold=./data/train_output_gold_1000.txt --save-to=./models/model_medium.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt --cuda
elif [ "$1" = "train_local" ]; then
	touch ./models/model.bin
	python run.py train --train-input=./data/train_small.txt --train-gold=./data/train_small_output_gold.txt \
        --batch-size=3 --valid-niter=100 --max-epoch=120 --save-to=./models/model.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt
elif [ "$1" = "train_local_2" ]; then
	touch ./models/model_2.bin
	python run.py train --train-input=./data/train_small.txt --train-gold=./data/train_small_output_gold.txt --batch-size=3 \
	--valid-niter=100 --max-epoch=120 --save-to=./models/model_2.bin --dev-input=./data/dev_small.txt --dev-gold=./data/dev_small_output_gold.txt
elif [ "$1" = "train_local_gru" ]; then
	touch ./models/model.bin
	python run.py train --train-input=./data/train_small.txt --train-gold=./data/train_small_output_gold.txt \
        --batch-size=3 --valid-niter=100 --max-epoch=120 --save-to=./models/model.bin --dev-input=./data/dev_small.txt\
         --dev-gold=./data/dev_small_output_gold.txt --model=gru_crf
elif [ "$1" = "test_local" ]; then
	mkdir -p outputs
	#rm -f outputs/test_local_outputs.txt
    #touch outputs/test_local_outputs.txt
    python run.py test --batch-size=1 --save-to=./models/model.bin --test-input=./data/dev_small.txt --test-gold=./data/dev_small_output_gold.txt --model-output=./outputs/test_local_outputs.txt 
elif [ "$1" = "test_local_gru" ]; then
	mkdir -p outputs
	#rm -f outputs/test_local_outputs.txt
    #touch outputs/test_local_outputs.txt
    python run.py test --batch-size=1 --save-to=./models/model.bin --test-input=./data/dev_small.txt --test-gold=./data/dev_small_output_gold.txt --model-output=./outputs/test_local_outputs.txt \
           --model=gru_crf
else
	echo "Invalid Option Selected"
fi
