# CS224N-PROJECT
Thai is one of the languages that does not have explicit segmentation, and cannot be used with most word based models. In this paper we will be tackling this problem by implementing BiLSTM-CRF and BiGRU-CRF based segmentation algorithms to parse Thai text. Our model achieves an F1 score of 94.78 (micro) and 96.26 (macro). Our model outperforms the micro averaged F1 score from previous models and has comparable macro F1 score. The model also works well on small data, but struggles with named entities.

# Model
![alt text](https://user-images.githubusercontent.com/32311654/54494702-5cc3a580-489a-11e9-94a6-f8d8f31fe66b.png)

# Results
Below is a summarized table of the character-level F1 metrics for our project
![alt text](https://user-images.githubusercontent.com/32311654/54494648-d73ff580-4899-11e9-8ac8-83fde9d58d79.png)
