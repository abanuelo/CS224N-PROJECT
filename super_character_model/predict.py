import tensorflow as tf
import numpy as np
import os,glob,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys,argparse
import csv
import re
import pandas as pd

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('quora-insincere-sincere-questions-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, len(os.listdir('training_data')))) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probability of sincere | probability that it is insincere]
#print(result)

#Gets whether the model reports sincere:0 or insincere: 1
prediction = 0
count = 0
first_item = 0
for element in result:
    for item in element:
        if count == 0:
            first_item = item
        else: 
            if item > first_item:
                prediction = 0
                if count == 1:
                    prediction = 1
        count += 1
        

#Gets the index of the question text we are looking at using regex
regex_searches = re.findall(r'\d+', image_path)
local_index = int(regex_searches[0])

#Gets the actual classification of that testing example
test_df = pd.read_csv("./test.csv", engine='python')
qu_id = test_df.qid

# #checks if we arrived at the right conclusion
# success = 0
# if prediction == actual:
#     success = 1
# else:
#     success = 0 

print("Model Probability Prediction: ", result)
print("Final Prediction: ", prediction)
print("Current Index: ", local_index)
print("---------------------------------------------")

#THIS NEXT PART IS GOING TO WRITE TO A CSV FILE
with open('submission.csv', mode='a') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow([qu_id[local_index+1], prediction])
