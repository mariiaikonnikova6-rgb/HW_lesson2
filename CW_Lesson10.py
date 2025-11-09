# Урок №10 - Створення власної моделі класифікації зображень (Scikit-Learn + OpenCV)

# встановити бібліотеку - pip install scikit-learn

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#1 створюємо функцію для генерації простих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8) #створюємо чисту матрицю (кадр) чорного кольору
    if shape == 'circle':
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == 'square':
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#2 Формуємо набір даних
X = []   # список ознак
y = []   # список міток

colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0)
}
shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):  # створюємо 10 прикладів кожного типу
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]  # повертає середнє (B,G,R,alpha) беремо перші 3.
            features = [mean_color[0], mean_color[1], mean_color[2]]  # ознаки тут — середній колір по каналах B, G, R:
            X.append(features) #усі ознаки (features), тобто числові дані, за якими навчається модель
            # в нашому випадку це одна з трьох фігур + значення кольору
            y.append(f"{color_name}_{shape}") #усі мітки (labels), тобто правильні відповіді, які модель повинна передбачати

#3 розділяємо дані 70% даних — для навчання, 30% — для перевірки
#У машинному навчанні ми ніколи не тренуємо модель на всіх даних одразу.
#Бо якщо вона “вивчить” усе напам’ять — ми не знатимемо, чи справді
#вона вміє узагальнювати, чи просто запам’ятала приклади.
#Тому ми ділимо весь набір даних (dataset) на дві частини:
#X_train, y_train — дані для навчання моделі, X_test, y_test — дані для перевірки (тестування)
#X_train -ознаки (features) для навчання, модель бачить їх і вчиться
#y_train - правильні відповіді (labels) для навчання, щоб модель знала, що є правильним
#X_test - ознаки для перевірки, нові дані, яких модель "не бачила"
#y_test	правильні відповіді для перевірки, щоб оцінити, наскільки модель вгадує

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
#stratify=y — зберігаємо однакову пропорцію класів у train і test (важливо, щоб оцінка була чесною).
#Без stratify можна випадково отримати дисбаланс: наприклад,
#у тесті з'являться майже лише квадрати, і метрика спотвориться.


#4 створюємо та навчаємо модель. Вона вчиться порівнювати об'єкти за схожістю кольорів
#k-Nearest Neighbors (KNN) — це дуже простий класифікатор:
# щоб визначити клас нового прикладу, він дивиться на k найближчих
#навчальних прикладів у просторі ознак і обирає більшість
model = KNeighborsClassifier(n_neighbors=3)#беремо 3 найближчих навчальних приклади
model.fit(X_train, y_train)#запам’ятали тренувальні приклади (і побудували структуру пошуку)

#4 перевірка точності
accuracy = model.score(X_test, y_test)#score для класифікатора — це accuracy частка вірних відповідей на тесті
print("Точність моделі:", round(accuracy * 100, 2), "%")

#5 тестуємо модель на новому зображенні
test_img = generate_image((0, 255, 0), 'circle')
mean_color = cv2.mean(test_img)[:3]
prediction = model.predict([mean_color])
print("Передбачення:", prediction[0])

# cv2.imshow("Test image", test_img)


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape
x = w//2 - 100
y = h//2 - 100
roi = frame[y:y+200, x:x+200]
# avg = cv2.mean(roi)
mean_color = cv2.mean(roi)[:3]
prediction = model.predict([mean_color])
print("Передбачення:", prediction[0])

cv2.imshow("Test image", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()