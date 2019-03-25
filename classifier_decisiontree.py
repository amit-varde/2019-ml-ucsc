# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
def processandtrainmodel():
	dataset = pd.read_csv('../data/TI-Opamps-Spec.csv')

	dataset['CMRR'].fillna(dataset['CMRR'].mean(),inplace=True)
	dataset['size'].fillna(dataset['size'].mean(),inplace=True)
	dataset['price'].fillna(dataset['price'].mean(),inplace=True)
	dataset['GBW'].fillna(dataset['GBW'].mean(),inplace=True)
	dataset['Package Group'].fillna('Default',inplace=True)

	filterDimNan = dataset['Dimension (mm)'].isna()
	datasetDimNan = dataset[filterDimNan]
	datasetNotDimNan = dataset[~filterDimNan]


	filterDimmm2 = datasetNotDimNan['Dimension (mm)'].str.contains('mm2')
	datasetDimMMe = datasetNotDimNan[filterDimmm2]

	datasetNotDimmm2 = datasetNotDimNan[~filterDimmm2]

	new = datasetNotDimmm2["Dimension (mm)"].str.split("x", n = 1, expand = True) 
	  
	# making seperate first name column from new data frame 
	datasetNotDimmm2["area"]= new[0].astype(float) * new[1].astype(float)
	datasetNotDimmm2.drop(columns =["Dimension (mm)"], inplace = True)
	#print(datasetNotDimmm2)

	newMM = datasetDimMMe["Dimension (mm)"].str.split(" ", n = 1, expand = True) 
	datasetDimMMe["area"]= newMM[0].astype(float) 
	datasetDimMMe.drop(columns =["Dimension (mm)"], inplace = True)
	#print(datasetDimMMe)

	meandataset = pd.merge(datasetNotDimmm2,datasetDimMMe, how='outer')
	#print("Mean --------  ", meandataset["area"].mean())


	datasetDimNan['Dimension (mm)'].fillna(meandataset["area"].mean(), inplace=True)
	#print("------------------------")

	datasetDimNan.rename(index=str,columns={"Dimension (mm)": "area"}, inplace=True)
	#print("datasetDimNan" ,datasetDimNan)
	finaldataset = pd.merge(meandataset,datasetDimNan, how='outer')

	#print("finaldataset",finaldataset)

	from sklearn import tree
	X = finaldataset.iloc[:, [1,2,3,4,5,6]].values
	y = finaldataset.iloc[:, 7].values

	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X,y)

	return clf

def predict(algo,specs):
	#print(specs)
	return algo.predict(specs)



