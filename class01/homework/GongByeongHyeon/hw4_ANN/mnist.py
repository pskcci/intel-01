import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(image_train, label_train,), (image_test, label_test) = mnist.load_data()
print("Train image shape : ", image_train.shape)
print("Traing Label : ", label_train, "\n")
print(image_train[0])

num = 20
for idx in range(num):
    sp = plt.subplot(5, 5, idx+1)
    plt.imshow(image_train[idx])
    plt.title(f'Label: {label_train[idx]}')
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28, 28)))  # 28 x 28의 이미지 파일
model.add(tf.keras.layers.Flatten())  # 한 줄로 정렬
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # 모델을 컴파일하여 최적화, 손실 함수, 평가 지표 정의
model.summary() 
model.fit(image_train, label_train, epochs=10, batch_size=10)  # 학습 실시
num = 3
predict = model.predict(image_test[0:num])
print(predict)
print(" prediction ", np.argmax(predict, axis=1))
plt.figure(figsize=(15, 15))
for idx in range(num):
    sp = plt.subplot(1, 3, idx+1)
    plt.imshow(image_test[idx])
plt.show()
