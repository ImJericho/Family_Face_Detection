import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import numpy as np
# importing the image to face.py
import image2face


'''Training,Testing and Labeling the Dataset into numpy array'''
y1 = [1 for i in range(15)]
y2 = [2 for i in range(15)]
y3 = [3 for i in range(15)]
y4 = [4 for i in range(15)]

y_train = np.array(y1+y2+y3+y4)
x = [[] for i in range(60)]

# reading >> gray scalling >> converting into (100,100px) >> flatning
def conversions(name):
    im = cv2.imread(name)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    img =cv2.resize(img,(150,150))
    flatenImage = img.flatten()

    return flatenImage


# importing and genrating numpy array
for i in range(15):
    name1 = 'Resources/only_faces/'+ str(image2face.file_name_1[i])
    name2 = 'Resources/only_faces/' + str(image2face.file_name_2[i])
    name3 = 'Resources/only_faces/' + str(image2face.file_name_3[i])
    name4 = 'Resources/only_faces/' + str(image2face.file_name_4[i])

    flatenImage1 = conversions(name1)
    flatenImage2 = conversions(name2)
    flatenImage3 = conversions(name3)
    flatenImage4 = conversions(name4)

    x[i] = list(flatenImage1)
    x[15+i] = list(flatenImage2)
    x[30+i] = list(flatenImage3)
    x[45 + i] = list(flatenImage4)


# converting list into array
x_train = np.array(x)
# shuffling the data points and label in same order
x_shuffled,y_shuffled = shuffle(x_train,y_train)


#////////////////////////////////////////////////////////////
''' TRAINING'''

mlp = MLPClassifier(hidden_layer_sizes=(1000,100),max_iter=100)
mlp.fit(x_shuffled,y_shuffled)



# NAMING THE DATASET
def name_pred(pred):
    if pred == 1:
        return 'Vivek'
    elif pred == 2:
        return 'Shivani'
    elif pred == 3:
        return 'shyam'
    elif pred == 4:
        return 'Rekha'




# '''TESTING'''
# predict = mlp.predict(flatenImage1.reshape(1,-1))
#
# print(predict)
#
# if predict == 1:
#     person = 'VIVEK'
# elif predict == 0:
#     person = 'SHIVANI'

# cv2.putText(im1,person,(50,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),thickness=3)
# cv2.imshow('test image',im1)
# cv2.waitKey(0)