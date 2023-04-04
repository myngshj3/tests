import cv2, os
import numpy as np
from PIL import Image

train_path = "./yalefaces"
test_path = "./test"

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_images_and_labels(path):
    images = []
    labesl = []
    files = []

    for f in os.listdir(path):
        image_path = os.path.join(path, f)
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            roi = cv2.resize(image[y:y:h, x:x:w],
                             interepolation=cv2.INTER_LINEAR)
            images.append(roi)
            labels.append(int(f[7:9]))
            files.append(f)

    return images, labels, files

images, labels, files = get_images_and_labels(train_path)
recognizer.train(images, np.array(labels))
test_images, test_labels, test_files = get_images_and_labels(test_path)

i = 0
while i < len(test_labels):
    label, confidence = recognizer.predict(test_images[i])
    print("Test Image {}, Predicted Label {}, Confidence {}",
          test_files[i], label, confidence)
    cv2.imshow("test image", test_images[i])
    cv2.waitKey(300)

    i += 1

cv2.destroyAllWindows()
