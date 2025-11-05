import cv2
import numpy as np
import pandas as pd


net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)      # ділимо тільки на 2 частини: id і назва
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


image1 = cv2.imread('images/MobileNet/fox.jpg')
image2 = cv2.imread('images/MobileNet/kenguru.jpg')
image3 = cv2.imread('images/MobileNet/panda.jpg')
image4 = cv2.imread('images/MobileNet/rat.jpg')
image5 = cv2.imread('images/MobileNet/pig.jpg')

image_names = ["fox.jpg", "kenguru.jpg", "panda.jpg", "rat.jpg", "pig.jpg"]
images = [image1, image2, image3, image4, image5]
blobs = []
tables = []

for i, image in enumerate(images):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )
    blobs.append(blob)

    net.setInput(blob)

    preds = net.forward()[0]
    top10_idx = np.argsort(preds)[-10:][::-1]
    top10_vals = preds[top10_idx].astype(float)

    tables.append([i] + top10_vals.tolist())


    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "unknown"
    conf = float(preds[0][idx]) * 100

    print("Клас:", label)
    print("Ймовірність:", round(conf, 2), "%")

    text = label + ": " + str(int(conf)) + "%"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(f'"result_{i}"', image)
    cv2.imwrite(f'"result_{i}.jpg"', image)


cols = ["image"] + [f"Top{k}" for k in range(1, 11)]
df = pd.DataFrame(tables, columns=cols)
df.to_csv("images/results_max1.csv", index=False)


cv2.waitKey(0)
cv2.destroyAllWindows()