import cv2
import numpy as np

# loop for the names
imgName_1 = []
imgName_2 = []
imgName_3 = []
imgName_4 = []

file_name_1 = []
file_name_2 = []
file_name_3 = []
file_name_4 = []

for i in range(15):
    # name_1 = 'Resources/original image/'+'IMG_20210521_22'+str(11+i)+'.jpg'
    # name_2 = 'Resources/original image/'+'IMG_20210521_2' +str(11+i)+'.jpg'
    # name_3 = 'original image/'+'IMG_face_3_ (' +str(i)+').jpg'
    name_4 = 'original image/' + 'IMG_face_4_ (' + str(i) + ').jpg'

    # imgName_1.append(name_1)
    # imgName_2.append(name_2)
    # imgName_3.append(name_3)
    imgName_4.append(name_4)

    # file_name_1.append('IMG_face_1_'+str(i)+'.jpg')
    # file_name_2.append('IMG_face_2_'+str(i)+'.jpg')
    # file_name_3.append('IMG_face_3_' + str(i) + '.jpg')
    file_name_4.append('IMG_face_4_' + str(i) + '.jpg')

if __name__ == '__main__':

    '''FOR SUBJECT_1'''
    j=0
    for i in range(15):

        img = cv2.imread(imgName_4[i])
        print(i)
        # img = cv2.resize(img,(800,800))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(img, 1.1, 4)

        for (x, y, w, h) in faces:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            imgCropped = img[y:y+h,x:x+w]
            # imgCropped = cv2.cvtColor(imgCropped,cv2.COLOR_BGR2GRAY)
            imgFinal = cv2.resize(imgCropped,(300,300))

            cv2.imshow('mummy', img)
            cv2.waitKey(10)

            '''run this'''
            file_name = 'IMG_face_4_'+str(i+j)+'.jpg'
            cv2.imwrite(file_name,imgFinal)
            j+=1