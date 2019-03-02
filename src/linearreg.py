#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split





def main():
    filename = '/Users/amit/ML/ClassProject/data/small.csv'
    #filename = '/Users/amit/ML/ClassProject/data/TI-Opamps-Spec.csv'
    dataset=pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    print ("Dataset Shape=\n", dataset.shape);
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=0)
    # Sperate train and test data
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=0)
    # There are three steps to model something with sklearn
    # 1. Set up the model
    model = LinearRegression()
    # 2. Use fit
    model.fit(X_train, y_train)
    # 3. Check the score
    model.score(X_test, y_test)


if __name__== "__main__":
    main()
