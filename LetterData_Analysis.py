import pandas as pd
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

#Loading the dataset
letterDataset = pd.read_csv("letter_recognition.data", header= None)
features = ["x-box", "y-box", "width", "height", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx", "letter", ]
target = ["letter"]
letterDataset.columns = features

#Scaling the attributes
scaler = MinMaxScaler()
for feature in features[:-1] :
      letterDataset[[feature]] = scaler.fit_transform(letterDataset[[feature]])

#Splitting the training and testing data
trainData = letterDataset[0:16000]
testData = letterDataset[16000:20000]

from sklearn import tree
from sklearn import ensemble

#Seperating the feature columns and target columns in the train and test data
targetTrain = trainData['letter']
del trainData['letter']
targetTest = testData['letter']
del testData['letter']

from sklearn.model_selection import cross_val_score

#Using cross validation finding the best Decision tree model
decisionTreeDepths = []
decisionTreeAcuracies = []
for depth in range(3,50):
      dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth,min_samples_leaf=5)
      scores = cross_val_score(dtc, trainData, targetTrain, cv=10, scoring='accuracy')
      decisionTreeDepths.append(depth)
      decisionTreeAcuracies.append(scores.mean())


import matplotlib.pyplot as plt

#Plotting the graph between various depths of decision trees and their accuracies
fig, ax = plt.subplots()
ax.set_xlim(min(decisionTreeDepths)-1, max(decisionTreeDepths)+1)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.plot(decisionTreeDepths,decisionTreeAcuracies, color ='g', lw=2)
plt.savefig("DecisionTreeLetterData.png")
plt.close()

#Building a final decision tree on the tuned parameters and making predicitions on the test dataset
finalDTC = tree.DecisionTreeClassifier(criterion='entropy', max_depth=18,min_samples_leaf=5)
finalDTC.fit(trainData, targetTrain)
expected = targetTest
predicted = finalDTC.predict(testData)
print "The best Accuracy for Decision Tree with tuned parameters", accuracy_score(expected, predicted)*100

#Using cross validation finding the best Random Forest classifier
rfDepth = []
rfAccuracies = []
for depth in range(10,100, 10):
      rfc = ensemble.RandomForestClassifier(n_estimators= depth, criterion= 'entropy', max_depth=18, bootstrap= True, min_samples_leaf=5)
      scores = cross_val_score(rfc, trainData, targetTrain, cv=10, scoring='accuracy')  
      rfDepth.append(depth)
      rfAccuracies.append(scores.mean())

#Plotting the graph between number of trees in a RandomForest model and their accuracies
fig, ax = plt.subplots()
ax.set_xlim(min(rfDepth)-1, max(rfDepth)+1)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")
plt.plot(rfDepth,rfAccuracies, color ='g', lw=2)
plt.savefig("RandomForestLetterData.png")
plt.close()

#Building a final Random Forest classifier on the tuned parameters and making predicitions on the test dataset
finalRF = ensemble.RandomForestClassifier(n_estimators= 80, criterion= 'entropy', max_depth=18, bootstrap= True, min_samples_leaf=5)
finalRF.fit(trainData, targetTrain)
expected = targetTest
predicted = finalRF.predict(testData)
print "The best Accuracy for Random Forest Classifier with tuned parameters", accuracy_score(expected, predicted)*100

from sklearn import svm

#Building a SVM model and making predictions on the test data
svmclf = svm.SVC(C=10, kernel='linear')
svmclf.fit(trainData, targetTrain)
expected = targetTest
predicted = svmclf.predict(testData)
print "The best Accuracy for SVM with tuned parameters", accuracy_score(expected, predicted)*100


from sklearn.neighbors import KNeighborsClassifier
#Using cross validation finding the best KNN classifier
kValues = []
kAccuracies = []
for k in range(2,10):
    neigh = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(neigh, trainData, targetTrain, cv=10, scoring='accuracy')
    kValues.append(k)
    kAccuracies.append(scores.mean())

#Plotting the graph between different k values and accuracies
fig, ax = plt.subplots()
ax.set_xlim(min(kValues)-1, max(kValues)+1)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(kValues,kAccuracies, color ='g', lw=2)
plt.savefig("kNNLetterData.png")
plt.close()

#Building a final KNN classifier on the tuned parameters and making predicitions on the test dataset
finalKMeans = KNeighborsClassifier(n_neighbors=3)
finalKMeans.fit(trainData, targetTrain)
expected = targetTest
predicted = finalKMeans.predict(testData)
print "The best Accuracy for KNN classifier with tuned parameters", accuracy_score(expected, predicted)*100

from sklearn.ensemble import AdaBoostClassifier

#Using cross validation finding the best Adaboost classifier
adaboostBaseEstimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10,min_samples_leaf=5)
adaNumEstimators = []
adaAccuracies = []
for estimators in range (10,100,10):
    adaclf = AdaBoostClassifier(base_estimator = adaboostBaseEstimator, n_estimators=estimators)
    scores = cross_val_score(adaclf, trainData, targetTrain, cv=10, scoring='accuracy')
    adaNumEstimators.append(estimators)
    adaAccuracies.append(scores.mean())

#Plotting the graph between different number of trees in the Adaboost classifier and accuracies
fig, ax = plt.subplots()
ax.set_xlim(min(adaNumEstimators)-1, max(adaNumEstimators)+1)
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.plot(adaNumEstimators,adaAccuracies, color ='g', lw=2)
plt.savefig("AdaBoostLetterData.png")
plt.close()

#Building a final Adaboost classifier on the tuned parameters and making predicitions on the test dataset
finalBaseDecisionTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10,min_samples_leaf=5)
finalAdaboost = AdaBoostClassifier(base_estimator = finalBaseDecisionTree, n_estimators=80)
finalAdaboost.fit(trainData, targetTrain)
expected = targetTest
predicted = finalAdaboost.predict(testData)
print "The best Accuracy for Adaboost with tuned parameters", accuracy_score(expected, predicted)*100