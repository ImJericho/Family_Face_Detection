import cv2
import numpy as np

# loop for the names
imgName_1 = []
imgName_2 = []

file_name_1 = []
file_name_2 = []
file_name_3 = []
file_name_4 = []

for i in range(15):
    name_1 = 'Resources/original image/'+'IMG_20210521_22'+str(11+i)+'.jpg'
    name_2 = 'Resources/original image/'+'IMG_20210521_2' +str(11+i)+'.jpg'
    imgName_1.append(name_1)
    imgName_2.append(name_2)

    file_name_1.append('IMG_face_1_'+str(i)+'.jpg')             #vivek
    file_name_2.append('IMG_face_2_'+str(i)+'.jpg')             #shivani
    file_name_3.append('IMG_face_3_' + str(i) + '.jpg')         #papa
    file_name_4.append('IMG_face_4_' + str(i) + '.jpg')         #mummy


if __name__ == '__main__':
    # importing the images from grayscale folder and saving only face
    # converting images into only faces

    '''FOR SUBJECT_1'''
    for i in range(15):

        im = cv2.imread(imgName_1[i])
        img = cv2.resize(im,(800,800))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier("Resources\haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(img, 1.1, 4)


        for (x, y, w, h) in faces:

            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # cv2.circle(img,(x,y),5,(0,255,0),thickness = 2)
            # cv2.line(img,(x,y),(x+w,y),(0,0,255),thickness=3)
            # cv2.line(img,(x,y),(x,y+h),(0,0,255),thickness=3)
            # print(w,h)

            imgCropped = img[y:y+h,x:x+w]
            imgFinal = cv2.resize(imgCropped,(300,300))

            # cv2.imshow('vivek',imgFinal)
            # cv2.waitKey(20)

            '''run this'''
            # cv2.imwrite(file_name_1[i],imgFinal)

    # '''FOR SUBJECT_2'''
    for i in range(15):

        im = cv2.imread(imgName_2[i])
        img = cv2.resize(im, (800, 800))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier("Resources\haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            imgCropped = img[y:y + h, x:x + w]
            imgFinal = cv2.resize(imgCropped, (300, 300))

            # cv2.imshow('shivani', imgFinal)
            # cv2.waitKey(200)

            '''run this'''
            # cv2.imwrite(file_name_2[i],imgFinal)



