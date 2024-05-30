import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

#name = input("enter your name:")

nic_images_path =[r'C:\Users\lab_services_student\Documents\python code\ICE\facialReg\nicolas1.png', r'C:\Users\lab_services_student\Documents\python code\ICE\facialReg\nicolas2.png']
test_img_path = r'C:\Users\lab_services_student\Documents\python code\ICE\facialReg\nicolas3.png'
output_img_path = 'nicolas5.jpg'

# Get images and labels for training
sample_data = []
ids = []
face_cascade =cv2.CascadeClassifier( r'C:\Users\lab_services_student\Documents\python code\ICE\haarcascade_frontalface_default.xml')

for images_path in nic_images_path:
    img = cv2.imread(images_path)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale,1.2,4)
    for (x, y, w, h) in faces:
        face_img = gray_scale[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        sample_data.append(face_img.flatten())
        ids.append("Nicolas Cage")

# Training the KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(sample_data,ids)

#  Loading and processing of the image
test_img = cv2.imread(test_img_path)
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(test_gray,1.2,4)

# test detection and labeling
for (x, y, w, h) in faces:
    face_img = gray_scale[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (100, 100))
    identify = model.predict([face_img.flatten()])
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 4)
    cv2.putText(test_img, identify[0], (x, y-9), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,50))
    
# Result saving
cv2.imwrite(output_img_path,test_img)

cv2.imshow('validate Face', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()