# Name: Cynthia Nosiri
# Class: EEGR 565.M85 - Machine Learning Applications
# Program to vectorize the body of each article using a max of 500 features, 
# get the frequency and plot a histogram
# Google drive link: https://drive.google.com/drive/u/1/folders/1USLN7EZnCJfW6LBxrmzTbVIVh2FG6Qxv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import feature_extraction

df = pd.read_csv('cnn_data_4_5.csv')
df2 = df.loc[:, "body"]

# convert the text to a matrix of token/word counts
vectorizer = feature_extraction.text.CountVectorizer(
    stop_words="english", max_features=500)
bag_of_words = vectorizer.fit_transform(df2)

# computes the sum over the rows giving the total for the body column
sum_words = bag_of_words.sum(axis=0)

#returns key value pair of words and its occurrence
words_freq = [(word, sum_words[0, idx])
              for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# prints out the words and their occurrences
for word, count in words_freq:
    print(word + ":", count)
words = []
freqs = []
for word, count in words_freq:
    words.append(word)
    freqs.append(count)

# plots the first 10 words and their frequencies
plt.bar(np.arange(10), freqs[:10], align='center')
plt.xticks(np.arange(10), words[:10])
plt.ylabel('Frequency')
plt.title('Top 10 Words')
plt.show()
