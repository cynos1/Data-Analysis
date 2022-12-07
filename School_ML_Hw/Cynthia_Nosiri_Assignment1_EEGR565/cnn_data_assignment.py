# Name: Cynthia Nosiri
# Class: EEGR 565.M85 - Machine Learning Applications
# Program to print the first 5 rows of the given csv file
# Google drive link: https://drive.google.com/drive/u/1/folders/1USLN7EZnCJfW6LBxrmzTbVIVh2FG6Qxv


import pandas as pd

df = pd.read_csv('cnn_data_4_5.csv')
df1 = df.head()

print(df1)





