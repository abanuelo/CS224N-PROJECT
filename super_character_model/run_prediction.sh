for i in $(seq 0 56370); do
    python ./predict.py ./testing_data/example_"$i".png
done