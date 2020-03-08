import time 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk 
import io 
import unicodedata 
import numpy as np 
import re 
import string 
from numpy import linalg 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.tokenize import PunktSentenceTokenizer 
from nltk.corpus import webtext 
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer 
import requests 
import os, sys
import json
from nltk import tokenize

def sentimentalAnalysis(sentences):
    res = {}
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            res[k] = ss[k]
    return res