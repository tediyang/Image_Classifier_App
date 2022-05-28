import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from PIL import Image

hub = hub
image_size = 224
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    
    imag = Image.open(image_path)
    converted_image = np.asarray(imag)
    processed_image = process_image(converted_image)
    image = np.expand_dims(processed_image, axis=0)
    prob = model.predict(image)
    prob = prob[0].tolist()
    probs = sorted(prob)[::-1][:top_k]
    
    classes = [str(prob.index(p)+1) for p in probs]
        
    return probs, classes