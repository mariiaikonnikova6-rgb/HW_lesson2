# Урок 9: Класифікація одного зображення за допомогою MobileNet (OpenCV DNN)
# Потрібні файли поруч із цим скриптом:
#  - mobilenet_deploy.prototxt   (архітектура мережі)
#  - mobilenet.caffemodel        (ваги мережі)
#  - synset.txt                  (список з 1000 назв класів ImageNet)
#  - images.jpg                     (будь-яке тестове фото)

import cv2

# 1) Завантажуємо попередньо навчену модель MobileNet (формат Caffe)
#    readNetFromCaffe() приймає два файли: опис шарів (prototxt) і ваги (caffemodel).
net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

# 2) Читаємо список назв класів із synset.txt
#    У типовому synset рядок має вигляд: "n01440764 tench, Tinca tinca"
#    Ми беремо все після першого пробілу — це і є людська назва класу.
classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)      # ділимо тільки на 2 частини: id і назва
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

# 3) Завантажуємо зображення з файлу
image = cv2.imread('images/cat.jpg')

# 4) Готуємо зображення для мережі: створюємо blob (тензор)
#    - змінюємо розмір на 224x224 (так очікує MobileNet)
#    - масштабуємо пікселі 1/127.5
#    - віднімаємо середні значення (127.5, 127.5, 127.5) для нормалізації
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)

# 5) Кладемо підготовлені дані в мережу і запускаємо "forward pass"
net.setInput(blob)
preds = net.forward()   # preds — це вектор ймовірностей для 1000 класів

# 6) Знаходимо індекс класу з найбільшою ймовірністю
idx = preds[0].argmax()

# 7) Дістаємо назву класу і впевненість (у відсотках)
label = classes[idx] if idx < len(classes) else "unknown"
conf = float(preds[0][idx]) * 100

# 8) Пишемо результат у консоль (для перевірки)
print("Клас:", label)
print("Ймовірність:", round(conf, 2), "%")

# 9) Підписуємо результат на зображенні простим текстом
text = label + ": " + str(int(conf)) + "%"
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 10) Показуємо вікно з результатом. Натисни будь-яку клавішу, щоб закрити.
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
