#!/usr/local/bin/python3

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
from nltk.tokenize import RegexpTokenizer

#import urllib2
#import html2text
# word stemmer
#stemmer = LancasterStemmer()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
#------------------------------------------------------------_#
#corpus_words = {}
#class_words = {}
stop_words = set(stopwords.words('english'))
BUILD_ENHANCED_DATASET = 0
#------------------------------------------------------------_#
def exact_search_match(filename, sentence):
  matches = []
  file = open(filename, 'r' ,encoding='utf-8')
  T = file.read()
  file.close()
  lines=[x.strip() for x in T.split('\n')] 
  for line in lines:
    m=line.split(',', 1)[0]
    #print ("Line = ", line)
    #print ("m =", m)
    for word in sentence.split():
      if line.lower().find(word.lower()) > 0:
        matches.append(m)
      #print ("\tWord =" ,word)
      #print ("\t match=",line.find(word))
  return list(set(matches))
  
#------------------------------------------------------------_#
def classify(sentence,class_words,corpus_words):
    high_class = None
    high_score = 0
    # loop through our classes
    scores={}
    for c in class_words.keys():
        score = calculate_class_score(sentence, c, class_words)
        if score > 0:
          scores[c]=score
    return list(set(scores.keys()))
    #return sorted(scores.keys())
#------------------------------------------------------------_#
def calculate_class_score(sentence, class_name, class_words):
    score = 0
    #print ("sentence=",sentence ," class_name=",class_name);
    #print ("\tclass_words=",class_words[class_name]);
    for word_tokens in nltk.word_tokenize(sentence):
        #print ("\tword_okens=",word_tokens.lower())
        if stemmer.stem(word_tokens.lower()) in class_words[class_name]:
            score += 1
    #print ("\tscore =", score)
    return score
#------------------------------------------------------------_#
def create_training_data(filename):
  class_words = {}
  corpus_words = {}
  file = open(filename, 'r',encoding='utf-8')
  T = file.read()
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
  for data in training_data:
    #for word in nltk.word_tokenize(data['desc']):
    for word in re.split('-|,| ',data['desc']):
        #word=re.sub(r'\d|-', ' ', word)
        if word not in ["?", "," ,"'s", ">", ")", "/", "(",'\d'] and word not in stop_words :
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            class_words[data['class']].extend([stemmed_word])
  return len(Item),class_words,corpus_words
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
  print ("Data Shape=", data.shape)
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
# Showing some examples of matching
def main():
  file= '../data/TI-Opamps.Desc.csv'
  if BUILD_ENHANCED_DATASET > 0:
    enh_file = os.path.splitext(file)[0] + '.enhanced.csv'
    print ("Descriptions file =" + file)
    print ("Enhanced Descriptions =" + enh_file)
    if os.path.isfile(enh_file):
      filename = file
    else: 
      buildDataSet(file,enh_file)
      filename = enh_file
  filename= '../data/TI-Opamps.Desc-head.csv'
  

  print ("Creating training data model from file ", filename)
  total_parts,class_words, corpus_words=create_training_data(filename)
  #print ("Corpus words and counts: \n" )
  #print ("Class words: %s" % class_words) 
  
  Sentences=[ "low-quiescent current amplifiers high-input impedance" ,"All Purposet", "all-purpose" , "unity gain-bandwidth",  "current amplifier", "I want to amplify current" , "amplify voltage " , "I want to amplify voltage" ];
  #Sentences=[ "all purpose" ]
  #Sentences=[ "precision " ]
  results = dict(); 
  for sentence in Sentences:
    words = re.split(r'\W+', sentence)
    filtered_words=[word.lower() for word in words if word.isalpha()]
    filtered_sentence = ' '.join(filtered_words)
    matches=classify(sentence,class_words,corpus_words)
    results[sentence]=matches
    print ("\nSentence: \'" +  sentence + "\'\nIs filerted to:\'" + filtered_sentence + "\'\nMatches=" + str(len(matches)) + " out of " + str(total_parts) + "\n NLP Matches=" + (str(matches)))
    exact_search_matches = exact_search_match(filename,sentence)
    print (" Exact Match search:", str(exact_search_matches))

#------------------------------------------------------------_#




if __name__== "__main__":
    main()
