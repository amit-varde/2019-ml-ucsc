import re
import linearreg
import classifier_decisiontree
import numpy as np
import nlreco
import os
import pandas as pd
import aggregator as agg

#clf = classifier_decisiontree.processandtrainmodel()

file= '../data/TI-Opamps.Desc.csv'
enh_file = os.path.splitext(file)[0] + '.enhanced_1.csv'
sales_file = os.path.splitext(file)[0] + '.sales.csv'
dataset = pd.read_csv(file)
class_words = {}
total_training_lines , class_words, corpus_words = nlreco.create_training_data(file)
while 1 > 0:
	searchString = input("Enter the search text: ")
	stringarray = re.split(":", searchString)
	description = stringarray.pop(0)
	if searchString == "exit":
		break
	else:
		spec = np.ndarray(shape=(6,1), dtype=float);
		i = 0
		for x in stringarray:
			spec.itemset(i,x)
			i = i+1

#		matches_specs = classifier_decisiontree.predict(clf,spec.reshape(1,-1))
		print("-----------------------------------------------")
		print("Specs Matches:", matches_specs)
		print("-----------------------------------------------")
		matches_desc = nlreco.classify(description, class_words, corpus_words)
		print("-----------------------------------------------")
		print("Description Matches:", matches_desc)
		print("-----------------------------------------------")
		agg.aggregate(matches_specs, matches_desc, dataset)
		#print(matches_desc)
		#print(len(matches_desc))

