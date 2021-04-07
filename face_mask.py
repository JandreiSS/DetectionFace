import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

detectorFace = cv.CascadeClassifier('FaceAndMaskDetection\haarcascade_frontalface_default.xml')

while(True):
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    deteccoes = detectorFace.detectMultiScale(gray, scaleFactor = 1.2, minSize = (100, 100))
    for (x, y, w, h) in deteccoes:
      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()