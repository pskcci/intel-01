import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#handwriten digit
mnist = tf.keras.datasets.fashion_mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()

print("Training data shape: ", image_train.shape, label_train.shape)
print("Train Labe : ", label_train, "\n")
print(image_train[0])

NUM = 20
plt.figure(figsize=(15,15))
for idx in range(NUM):
    sp = plt.subplot(5,5,idx+1)
    plt.imshow(image_train[idx])
    plt.title(f'Label: {label_train[idx]}')
plt.show()