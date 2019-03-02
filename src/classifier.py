#!/usr/local/bin/python
# Support Vector Machine (SVM)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Classifiers imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

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




#-----------------------------------------------------------------------------#


# Importing the dataset
dataset=pd.read_csv('/Users/amit/ML/ClassProject/data/TI-Opamps-Spec.csv')
#dataset=pd.read_csv('/Users/amit/ML/ClassProject/data/small.csv')
dataset.fillna(0, inplace=True)

#print(dataset)

target_names=dataset.columns
print("Columns for the model")
print (target_names)
product_names = dataset.iloc[:, 1].values
print("Product Names")
print (product_names)

#X = dataset.iloc[:, [1,2, 3,4,5]].values
X = dataset.iloc[:, [4,5]].values
y = dataset.iloc[:, 7].values

print " Shape of X=" , X.shape
print " Shape of y=", y.shape
# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print "Shapes "
print "Train X=", X_train.shape , " y=", y_train.shape
print "Test  X=", X_test.shape , " cy=", y_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ====> INSERT YOUR CLASSIFIER CODE HERE <====
#forest-classifier.py:
#classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state = 0)

#knn-classifier.py:
#classifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p=2)

#naive-bayes-classifier.py:
#classifier = GaussianNB()

#svm-classifier.py:
#classifier = SVC(kernel = 'linear', random_state = 0)

#tree-classifier.py:
classifier = DecisionTreeClassifier(criterion='entropy', random_state = 0)


model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# ====> INSERT YOUR CLASSIFIER CODE HERE <====
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

print "True y=" , y_test
print "Pred y=" , y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("confusion matrix:")
print(cm)
plot_confusion_matrix(cm,classes=opamps)
