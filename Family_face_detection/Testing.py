import cv2
import numpy as np

# geting the previosly fetched training data
import Training

# to detect face not recognize
faceCascade = cv2.CascadeClassifier("E:\OPEN_CV\Resources\haarcascade_frontalface_default.xml")

# to capture live video from webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

# running the loop

while True:
    sccess, frame = cap.read()
    faces = faceCascade.detectMultiScale(frame,1.1,4)

    cropped_img =np.array([])
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cropped_img = frame[y:y+h,x:x+w]                        #only face image

        cropped_img = cv2.resize(cropped_img,(150,150))
        cropped_gray_img = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
        # creat a 1-D array
        flatten_img = cropped_gray_img.flatten()

        Prediction = Training.mlp.predict(flatten_img.reshape(1,-1))

        # naming the person in the image
        person = Training.name_pred(Prediction)
        # COUSTMISING THE WEBCAM WINDOW
        cv2.putText(frame,person,(x+w//2-10,y+h+20),cv2.FONT_HERSHEY_PLAIN,2,(0,255,25),thickness=2)


    # show and exit from the webcam window
    cv2.imshow('video',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break