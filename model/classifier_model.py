import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('test_images/sharapova1.jpeg')
#print(img.shape) 3D shape (x,y RGB values)
#plt.imshow(img) to show image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(gray.shape)

#detecting features of the picture
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)


