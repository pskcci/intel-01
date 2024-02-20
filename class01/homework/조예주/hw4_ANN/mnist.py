import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(image_train, lable_train), (image_test, label_test) = mnist.load_data()
print("train image shape : ", image_train.shape)
print("train label : ", lable_train.shape, "\n")
print(image_train[0])

#20개의 handwritten 숫자 출력
num = 20
plt.figure(figsize=(15,15))
for idx in range (num):
    sp = plt.subplot(5,5,idx+1)
    plt.imshow(image_train[idx])
    plt.title(f'Label: {lable_train[idx]}')
plt.show()


#
model = tf.keras.Sequential()                      # model 이름으로 만듦
model.add(tf.keras.Input(shape=(28,28)))           # input 크기부터 정의
model.add(tf.keras.layers.Flatten())                      # Flatten (28x28=784개로 넓히기)
model.add(tf.keras.layers.Dense(128,\
                         activation = 'sigmoid'))          # 일단 128개로 줄이기, sigmoid함수로 활성화
model.add(tf.keras.layers.Dense(64,\
                         activation = 'sigmoid'))          # 일단 64개로 줄이기, sigmoid함수로 활성화   
model.add(tf.keras.layers.Dense(10,\
                         activation = 'softmax'))          # 최종 10개로 줄이기, softmax함수로 활성화

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#모델링
model.fit(image_train, lable_train, epochs=10, batch_size=10)   #트레이닝 데이터셋 불러와서 학습시키기

#이미지 3장으로 하기
num = 3
predict = model.predict(image_test[0:num])
print(predict)                       #확률값들이 나옴
print("*prediction, ", np.argmax(predict, axis=1))
plt.figure(figsize = (15,15))
for idx in range(num):
    sp =plt.subplot(1,3,idx+1)
    plt.imshow(image_test[idx])
plt.show()