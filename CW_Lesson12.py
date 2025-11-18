import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras.models import image

train_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/train',
                                                               image_size=(128, 128),
                                                               batch_size=30, #кількість фото за один крок навчання
                                                               label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/test',
                                                               image_size=(128, 128),
                                                               batch_size=30, #кількість фото за один крок навчання
                                                               label_mode='categorical')
#нормалізація зображень

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


#побудова моделі
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3))) #1 фільтр, найпростіші ознаки, лінії

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) #2 фільтр, контури, структури обёєктів

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu')) #3 фільтр, визначає фігури

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten()) #перетворили в вектори

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=15,
    validation_data=test_ds,
)

test_lost, test_acc = model.evaluate(test_ds)
print(f'Якість{test_acc}')

class_name = ['car', 'cats', 'dogs']

img = image.load_img('images/', target_size=(128, 128))

img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])
print(f'Імовірність по класам {predictions[0][predicted_index]}')
print(f'Модель визначила {class_name[predicted_index]}')
