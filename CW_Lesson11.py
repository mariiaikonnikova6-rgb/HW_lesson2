import pandas as pd #робота з csv
import numpy as np #математичні операції
import tensorflow as tf #для нейронок бібліотека
from keras import Sequential
from tensorflow import keras # для тенсор
from tensorflow.keras import layers #для створення шарів
from sklearn.preprocessing import LabelEncoder #перетворює текстові мітки в числа
import matplotlib.pyplot as plt #для побудови графіків

#2 робота з csv
df = pd.read_csv('data/figures.csv')
#print(df.head())

#3 обираємо елементи для навчання
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

#4 створємо модель
model = keras.Sequential([layers.Dense(1, activation='relu', input_shape=(3,)),
                          layers.Dense(1, activation='relu', input_shape=(3,)),
                          layers.Dense(1, activation='softmax')])
print(type(model))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#5 навчання
history = model.fit(X, y, epochs = 200, verbose = 0)

#6 візуалізація навчання
plt.plot(history.history['loss'], label = 'Втрата (Loss)')
plt.plot(history.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання')
plt.legend()
plt.show()


#6 тестування
test = np.array([[18, 16, 3]])

pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'модель визначила: {encoder.inverse_transform(pred)}')


