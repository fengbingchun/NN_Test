# reference: https://docs.opencv.org/3.1.0/d5/d26/tutorial_py_knn_understanding.html
#            opencv-3.3.0/doc/py_tutorials/py_ml/py_knn/py_knn_opencv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Blog: http://blog.csdn.net/fengbingchun/article/details/78465497

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25,2)).astype(np.float32)
# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel() == 0]
plt.scatter(red[:,0], red[:,1], 80, 'r', '^')

# Take Blue families and plot them
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')

#plt.show()

# New comer is marked in green color
newcomer = np.random.randint(0, 100, (1,2)).astype(np.float32)
plt.scatter(newcomer[:,0], newcomer[:,1], 80, 'g', 'o')

knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print("result: ", results,"\n")
print("neighbours: ", neighbours,"\n")
print("distance: ", dist)

plt.show()