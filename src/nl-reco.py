#!/usr/local/bin/python

# Some basic ML modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sys
import re
import os

# NL modules
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import urllib2
import html2text
# word stemmer
#stemmer = LancasterStemmer()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
#------------------------------------------------------------_#
corpus_words = {}
class_words = {}
stop_words = set(stopwords.words('english'))
#------------------------------------------------------------_#
def exact_search_match(filename, s1, s2):
  file = open(filename, 'rt')
  T = file.read().decode("utf8")
  file.close()
  Item=[x.strip() for x in T.split('\n')] 
  matches = []
  for i in Item:
    w=re.split(r'\W+', i)
    if s1 in w and s2 in w:
      matches.append(w[1])
  return matches
#------------------------------------------------------------_#
def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    scores={}
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest score
        if score > 0:
          scores[c]=score
    return sorted(scores.keys())
    #return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
#------------------------------------------------------------_#
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    for word_tokens in nltk.word_tokenize(sentence):
        word  = ' '.join([w for w in word_tokens if not w in stop_words])
        if stemmer.stem(word_tokens.lower()) in class_words[class_name]:
            score += 1
    return score
#------------------------------------------------------------_#
def create_training_data(filename):
  file = open(filename, 'rt')
  T = file.read().decode("utf8")
  file.close()
  Item=[x.strip() for x in T.split('\n')] 
  training_data=[]
  # Building traing_data_set
  for i in Item:
    w=re.split(r'\W+', i)
    training_data.append({"class":w[0],"desc":i})
  print ("%s sentences of training data" % len(training_data))
  classes = list(set([a['class'] for a in training_data]))
  for c in classes:
    class_words[c] = []   
  # loop through each sentence in our training data
  for data in training_data:
    for word in nltk.word_tokenize(data['desc']):
        if word not in ["?", "'s", ">", '\d']:
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            class_words[data['class']].extend([stemmed_word])
  return len(Item)
#------------------------------------------------------------_#
def page_desc(url):
  import os
  from bs4 import BeautifulSoup
  cmd = "curl " + url
  f=os.popen(cmd)
  rawhtml= "";
  for i in f.readlines():
    rawhtml = rawhtml + i
  soup = BeautifulSoup(rawhtml)
  for script in soup(["script", "style"]):
    script.extract()    # rip it out
  text = soup.get_text()
  lines = (line.strip() for line in text.splitlines())
  chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
  text = ' '.join(chunk for chunk in chunks if chunk)
  return text
#------------------------------------------------------------_#
def buildDataSet(filename, enh_filename):  
  data = pd.read_csv(filename) 
  print "Data Shape=", data.shape
  parts=data['Part Number']
  online_desc=[]
  vendor_url='http://www.ti.com/product/'
  for p in parts:
     part_url= vendor_url + p
     part_desc=page_desc(part_url)
     online_desc.append(part_desc)
  data['Online'] =online_desc
  data.to_csv(enh_filename, sep=',', encoding='utf-8')
  return enh_filename
#------------------------------------------------------------_#
def readSalesData(filename):  
  data = pd.read_csv(filename) 
  data = data.fillna(1.0)
  return data
#------------------------------------------------------------_#
def scoreBasedOnSalesData(sdata, list):
  return 0;
  

#------------------------------------------------------------_#
# Showing some examples of matching
def main():
  file= '../data/TI-Opamps.Desc.csv'
  enh_file = os.path.splitext(file)[0] + '.enhanced.csv'
  sales_file = os.path.splitext(file)[0] + '.sales.csv'
  
  print "Descriptions file =" + file
  print "Enhanced Descriptions =" + enh_file
  print "Sales Data =" + sales_file
  if os.path.isfile(enh_file):
    filename = file
  else: 
     buildDataSet(file,enh_file)
     filename = enh_file
  filename= '../data/TI-Opamps.Desc.csv'
  
#  exact_search_matches = exact_search_match(enh_file, "current" ,"amplifier")
#  print len(exact_search_matches)

  print "Creating training data model from file ", filename
  total_parts=create_training_data(filename)
  #print ("Corpus words and counts: %s \n" % corpus_words)
  #print ("Class words: %s" % class_words) 
  # Somthing that matches
  Sentences=[ "low-quiescent current amplifiers high-input impedance" ,"amit", "unity gain-bandwidth"]
  for sentence in Sentences:
    print "\nSentence = ", sentence
    matches=classify(sentence)
    print "Total Matches=" + str(len(matches)) + " out of " + str(total_parts) + " Matches"
    matched_parts=[]
    for m in matches:
      matched_parts.append(m[0]);
  
  Sentences=[ "current amplifier", "I want to amplify current" , "amplify voltage " , "I want to amplify voltage" ];
  results = dict(); 
  for sentence in Sentences:
    print "\nSentence = ", sentence
    matches=classify(sentence)
    print "Total Matches=" + str(len(matches)) + " out of " + str(total_parts) + " Matches"
    results[sentence]=matches
    print matches
    sys.exit()

  print "Does my recommendation change based on search words??" 
  print Sentences[0] , " vs ", Sentences[1] , " --> ", str(len(set(results[Sentences[0]]) - set(results[Sentences[1]])))
  print Sentences[0] , " vs ", Sentences[2] , " --> ", str(len(set(results[Sentences[0]]) - set(results[Sentences[2]])))
  print Sentences[1] , " vs ", Sentences[2] , " --> ", str(len(set(results[Sentences[1]]) - set(results[Sentences[2]])))
  print Sentences[2] , " vs ", Sentences[3] , " --> ", str(len(set(results[Sentences[2]]) - set(results[Sentences[3]])))
#------------------------------------------------------------_#




if __name__== "__main__":
    main()
