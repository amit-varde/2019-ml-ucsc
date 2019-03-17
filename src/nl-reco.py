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
import urllib2
import html2text
# word stemmer
#stemmer = LancasterStemmer()

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
#------------------------------------------------------------_#
corpus_words = {}
class_words = {}
#------------------------------------------------------------_#
def classify(sentence, match_type=0):
    # 1 = Print the best match

    high_class = None
    high_score = 0
    # loop through our classes
    scores={}
    
    for c in class_words.keys():
        # calculate score of sentence for each class
        #score = calculate_class_score_commonality(sentence, c, show_details=False)
        score = calculate_class_score(sentence, c, show_details=False)
        # keep track of highest score
        if score > 0:
          scores[c]=score
        if score > high_score:
          high_class = c
          high_score = score
    if high_score < 1:
      return [(high_class, high_score)]
    if match_type == 1:
      return [(high_class, high_score)]
      # Choosing not to return the sorted list - sorting may be expensive
      return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[0]
    else:
      return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    
#------------------------------------------------------------_#
def calculate_class_score(sentence, class_name, show_details=True):
    score = 0
    # tokenize each word in our new sentence
    for word in nltk.word_tokenize(sentence):
        # check to see if the stem of the word is in any of our classes
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treat each word with same weight
            score += 1
            if show_details:
                print ("   match: %s" % stemmer.stem(word.lower() ))
    return score
#------------------------------------------------------------_#
def create_training_data(filename):
  vendor_url='http://www.ti.com/product/'
  file = open(filename, 'rt')
  # T=FullText and decode to get rid of unicode
  #T = file.read().decode("utf8").replace("-"," ") 
  T = file.read().decode("utf8")
  file.close()
  Item=[x.strip() for x in T.split('\n')] 
  training_data=[]
  for i in Item:
    w=re.split(r'\W+', i)
    #part_url= vendor_url + w[0]
    #online_page_desc=page_desc(part_url);
    #part_info=i + online_page_desc
    #training_data.append({"class":w[0],"desc":part_info})
    training_data.append({"class":w[0],"desc":i})
  print ("%s sentences of training data" % len(training_data))
  classes = list(set([a['class'] for a in training_data]))
  for c in classes:
    class_words[c] = []   
  # loop through each sentence in our training data
  for data in training_data:
    # tokenize each sentence into words
    for word in nltk.word_tokenize(data['desc']):
        if word not in ["?", "'s", ]:
            # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            # add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])
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

#------------------------------------------------------------_#
# Showing some examples of matching
def main():
  #filename = '../data/small-desc.csv'
  file= '../data/TI-Opamps-Desc.csv'
  enh_file = os.path.splitext(file)[0] + '.enhanced.csv'
  if os.path.isfile(enh_file):
    filename = file
  else: 
     buildDataSet(file,enh_file)
     filename = enh_file

  print "Creating training data model from file ", filename
  create_training_data(filename)
  #print ("Corpus words and counts: %s \n" % corpus_words)
  #print ("Class words: %s" % class_words)
  
  # Somthing that matches
  sentence = "low-noise precision"
  for sentence in [ ", low-quiescent current amplifiers offer high-input impedance" , "low-noise" , "low-noise precision" ,  "amit" ]:
    print "\nSentence = ", sentence
    print  "Top Matchs =" 
    print classify(sentence,1)
    print  "All Match =" 
    print classify(sentence)
  
#------------------------------------------------------------_#




if __name__== "__main__":
    main()
