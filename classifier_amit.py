#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import sys
import re
import itertools
from itertools import product

#-----------------------------------------------------------------------------#
# Classifiers imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#-----------------------------------------------------------------------------#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#-----------------------------------------------------------------------------#
def setPartClassId(list): 
    unique_list = [] 
    for x in list: 
      if x not in unique_list: 
        unique_list.append(x) 
    return unique_list 
#-----------------------------------------------------------------------------#
# Importing the dataset
dataset=pd.read_csv('../data/TI-Opamps-Spec.csv')
#dataset=pd.read_csv('../data/TI-Opamps.Desc.sales.csv')


PClassID = [] 
P= dict()
PID=0
for p in dataset['Part Number']:
  pc=re.sub(r'\d|-', '', p)[0:2]
  if pc in P.keys():
    PClassID.append(P[pc])
  else:
    P[pc]= PID;
    PID = PID + 1;
    PClassID.append(P[pc])

dataset['PartsClassID']= PClassID

#print "PClassID=",PClassID
#print sorted(PClassID)
#print "# of Classes =", str(len(P))
#print "Classes= " , P

#feature_names=["price","Customer"]
#feature_names=["Channels","GBW","CMRR","size","price","Dimension (mm)","Customer"]
#feature_names=["Channels","GBW","CMRR","size"]
feature_names=["Channels","GBW","CMRR","size","price","Customer"]

#dataset[['Channels','GBW']]=dataset[['Channels','GBW']].fillna(value=1,inplace=True)
dataset['GBW'].fillna(dataset['GBW'].median(), inplace=True)
dataset['CMRR'].fillna(dataset['CMRR'].median(), inplace=True)
dataset['size'].fillna(dataset['size'].mean(), inplace=True)
dataset['price'].fillna(dataset['price'].mean(), inplace=True)
X = dataset[feature_names]
y = dataset['PartsClassID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#-----------------------------------------------------------------------------#
cList = dict()
cList['DecisionTreeClassifier']=DecisionTreeClassifier()
cList['RandomForestClassifier'] = RandomForestClassifier(n_estimators = 40, verbose=0, n_jobs=10, max_features = "log2", random_state = 7)
cList['KNeighboursClassifier'] = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2)
cList['Gaussian'] = GaussianNB()
cList['SVC']=SVC(kernel = 'linear', random_state = 0)


for c in sorted(cList.keys()):
  classifier = cList[c]
  classifier.fit(X_train, y_train)
  y_predict = classifier.predict(X_test)
  #cm = confusion_matrix(y_test, y_pred)
  print('Classifier ' + str(c) )
  #print ' on training set: {:.2f}'.format(classifier.score(X_train, y_train))
  #print ' on test set: {:.2f}'.format(classifier.score(X_test, y_test))
  print (" accuracy_score ", round(accuracy_score(y_test, y_predict)*100,2))

#sys.exit()
