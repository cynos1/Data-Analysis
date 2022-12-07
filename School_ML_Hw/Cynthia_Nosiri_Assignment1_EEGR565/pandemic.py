# Name: Cynthia Nosiri
# Class: EEGR 565.M85 - Machine Learning Applications
# Program to Filter out any tokens 
# that do not fall within the pandemic word list to display the relative frequency of terms related to the 
# pandemic used within the “Business” section of the CNN articles. 
# Google drive link: https://drive.google.com/drive/u/1/folders/1USLN7EZnCJfW6LBxrmzTbVIVh2FG6Qxv


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
              idx in vectorizer.vocabulary_.items() if sum_words[0, idx] < 1000]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# ---Read the txt file
with open('pandemic.txt') as f:
    lines = f.readlines()
    # print(lines)

# ---the text file comes with \n , so we strip it from the text file
pandemic = [line.strip("\n") for line in lines]
# print(pandemic)

# ---take out the title "Terms and splits into individual words"
pandemic_list = []
for a in [x.split(' ') for x in pandemic[1:]]:
    for b in a:
        pandemic_list.append(b.lower())
# print(pandemic_list)

 # -----split words with hyphens
new_pandemic_list = []
for a in pandemic_list:
    for b in (a.split('-')):
        new_pandemic_list.append(b)
# print(pandemic_lists)

# ----print all words (with their occurrences) in pandemic list contained in our original words list
for i in words_freq:
    if i[0] in new_pandemic_list:
        print(i)

# ----get the first 10 words for plotting
top_10 = pd.DataFrame([a for a in words_freq if a[0] in new_pandemic_list], columns=[
                      'words', 'frequency'])[:10]
# print(top_10)

# ----plot
plt.bar(top_10['words'], top_10['frequency'])
plt.ylabel('Frequency')
plt.title('Top 10 Words')
plt.show()
