#!/usr/local/bin/python3
from flask import Flask, render_template, request, flash

from flask_wtf import Form
from wtforms import TextField, SubmitField
from wtforms import validators, ValidationError

import subprocess
import re
import pandas as pd

# ML Libraries
import nlreco
import classifier
import aggregator as agg


def runcmd():
  p = subprocess.Popen("date", stdout=subprocess.PIPE, shell=True)
  (output, err) = p.communicate()
  p_status = p.wait()
  print ("Command output : ", output)
  #print "Command exit status/return code : ", p_status
#-------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------------#
class searchForm(Form):
  search = TextField("Search", [ validators.Required("Please enter search string") ])
  submit = SubmitField("send")
#-------------------------------------------------------------------------------------#
# create the application object
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'


# use decorators to link the function to a url
#-------------------------------------------------------------------------------------#
@app.route('/')
def home():
  form = searchForm();
  if request.method == 'POST':
    if form.validate == False: 
      flash('Some text is required!')
      return render_template('google.html', form = form)
    else:
      return render_template('google.html', form = form)
  return render_template('google.html', form = form)
#-------------------------------------------------------------------------------------#
@app.route('/reco',methods=['POST'])
def reco():
  filename_desc= '../data/TI-Opamps.Desc.csv'
  filename_spec= '../data/TI-Opamps-Spec.csv'

  dataset_nlp = pd.read_csv(filename_desc)
  dataset_agg = pd.read_csv(filename_spec)

  form = searchForm();
  search_string= request.form.get("search") 
  output = "Search String is " + str(search_string)
  string_array = re.split(":",search_string)
  
  nlp_input=string_array.pop(0)
  spec=np.ndarray(shape=(4,1),dtype=float)
  if len(string_array) == 4:
    i=0
    for x in string_array:
      spec.itemset(i,x)
      i=i+1
  else:
    # Some random stuff to get some result
    # OPA2356-EP,2,450,80,15,3.83,4.9 x 3,6
    spec.itemset(0,2)
    spec.itemset(1,450)
    spec.itemset(2,80)
    spec.itemset(3,15)
  # Classifier 
  clf, dataset_specs = classifier.processandtrainmodel()
  matches_specs = classifier.predict(clf,spec.reshape(1,-1), dataset_specs) 
  print(matches_specs)
   # NLP Recommendation 
  total_parts,class_words, corpus_words=nlreco.create_training_data(filename_desc)
  words = re.split(r'\W+', search_string)
  filtered_words=[word.lower() for word in words if word.isalpha()]
  filtered_sentence = ' '.join(filtered_words)
  matches_nlp=nlreco.classify(filtered_sentence,class_words,corpus_words)
  # Aggregating
  matches_agg=agg.aggregate(matches_specs, matches_nlp, dataset_agg)
  print ("----")
  matches=matches_agg['Part Number'].to_string().split(' ')
  print(output)

  print ("----")
  header=output.split('\n')
  return render_template('reco.html', output=header, matches_specs=matches_specs, matches_nlp=matches_nlp,matches =matches)  # render a template
  


#-------------------------------------------------------------------------------------#
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')  # render a template
#-------------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run(debug=True)
