# Name: Cynthia Nosiri
# Class: EEGR 565.M85 - Machine Learning Applications
# Program to filter out any token with a value larger than 1000.
# Google drive link: https://drive.google.com/drive/u/1/folders/1USLN7EZnCJfW6LBxrmzTbVIVh2FG6Qxv

# code is the same with vectorization.py with one exception
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import feature_extraction

df = pd.read_csv('cnn_data_4_5.csv')
df2 = df.loc[:, "body"]

vectorizer = feature_extraction.text.CountVectorizer(
    stop_words="english", max_features=500)
bag_of_words = vectorizer.fit_transform(df2)

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word,
              idx in vectorizer.vocabulary_.items() if sum_words[0, idx] < 1000] #right here: keeping only words with values less than 1000
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
for word, count in words_freq:
    print(word + ":", count)

words = []
freqs = []

for word, count in words_freq:
    words.append(word)
    freqs.append(count)

plt.bar(np.arange(10), freqs[:10], align='center')
plt.xticks(np.arange(10), words[:10])
plt.ylabel('Frequency')
plt.title('Top 10 Words')
plt.show()
