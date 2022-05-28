import argparse
import tensorflow as tf
from conversion import *
import json


parser = argparse.ArgumentParser(usage= 'predict.py ../image_path saved_model', description='This app classifies various flowers.', prog="Image Classifier")
parser.add_argument('image_path', default=None, help= 'the path the image is saved')
parser.add_argument('saved_model', default = 'my_class_model.h5', help= 'the saved model')
parser.add_argument('--top_k', type= int, default= 3, dest= 'top_k', help='the number of probabilities for the class, if none is given the default is 3.')
parser.add_argument('--category_names', dest= 'cat_nam', help='the file which consists of the names')

args = parser.parse_args()

model = args.saved_model

my_model = tf.keras.models.load_model(f'./{model}', custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
image_path = args.image_path
top_k = args.top_k
load_json = args.cat_nam

probs, classes = predict(image_path, my_model, top_k)
if load_json != None:
    with open(load_json, 'r') as f:
        class_names = json.load(f)
    prob_class_names = [class_names[n] for n in classes]
    print(probs)
    print(f'The flowers is/are: {prob_class_names}' )
    
else:
    print(probs)
    