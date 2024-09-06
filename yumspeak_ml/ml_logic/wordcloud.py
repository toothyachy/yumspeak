import pandas as pd
import string
import json
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from yumspeak_ml.params import *


# Set stop words
stop_words = set(stopwords.words('english'))
# Add our custom stopwords
custom_stopwords = set([word[0] for word in STOPWORDS])
stop_words.update(custom_stopwords)

def replace_punctuation_with_space(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

def data_processing(metadata):
    for i in metadata:
        # Remove numbers
        tokens = [''.join(char for char in review if not char.isdigit()) for review in i['review_text']]
        # Remove punctuation by replacing it with spaces
        tokens = [replace_punctuation_with_space(review) for review in tokens]
        # Strip trailing spaces
        tokens = [review.strip() for review in tokens]
        # Convert to lowercase and then tokenize each sentence
        tokens = [word_tokenize(review.lower()) for review in tokens]
        # Flatten the list
        tokens = [word for review in tokens for word in review]
        # Remove stop words
        tokens = [w for w in tokens if not w in stop_words]
    return tokens
