import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()

print("Train Image shape :",image_train.shape)
print("Train Labe : ", label_train,"\n")
print(image_train[0])

num = 20
plt.figure(figsize=(15,15))
for idx in range (num):
    sp = plt.subplot(5,5,idx+1)
    plt.imshow(image_train[idx])
    plt.title(f'Label: {label_train[idx]}')
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28, 28))) #인풋 사이즈
model.add(tf.keras.layers.Flatten()) # 플렛 평평히 펴주기 플렛트을 할 경우 원본 형상이 날라갈 수 있음 (굴곡 등등) 
model.add(tf.keras.layers.Dense(64, activation='sigmoid')) # 줄이기
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 줄이기

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(image_train, label_train, epochs = 10, batch_size=10) #트레이닝

num = 3
predic = model.predict(image_test[0:num])
print(predic)

print( "* prediction, ", np.argmax(predic, axis = 1))
plt.figure(figsize = (15, 15))
for idx in range(num):
    sp = plt.subplot(1, 3, idx+1)
    plt.imshow(image_test[idx])

plt.show()