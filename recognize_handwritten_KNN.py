# import the necessary packages
from __future__ import print_function
# import from the sklearn library
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
# import from skimage library
from skimage import exposure
# import from numpy library
import numpy as np
import matplotlib.pyplot as plt

# load in the digits dataset
digits_data = datasets.load_digits()

#train-test split
(trainData, testData, trainVal, testVal) = train_test_split(digits_data.data, digits_data.target,test_size = 0.25, random_state = 42)

#train-validation split 
(trainData, valiData, trainVal, valiVal) = train_test_split(trainData, trainVal, test_size = 0.1, random_state = 77)

#show the size of each train test split
print("The number of training", len(trainData))
print("The number of validation", len(valiData))
print("The number of testing", len(testData))

#Initiate an array to experiment on the K parameter
kValues = range(1,30,1)

#Initiate an array to store accuracy
accu = []

#Test through all the k values
for k in kValues:
	model = KNeighborsClassifier(n_neighbors = k)
	model.fit(trainData, trainVal)
	score = model.score(valiData,valiVal)
	print("K value is: ", k, " Score:", score)
	accu.append(score)


#best performance one
k_chosen = np.argmax(accu)

#re-train everything on the training data and test on the testing dataset
model = KNeighborsClassifier(n_neighbors = kValues[k_chosen])
model.fit(trainData, trainVal)
predictions = model.predict(testData)

#Conlusion
print("Evaluation on the model")
print(classification_report(testVal, predictions))
