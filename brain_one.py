# -*- coding: utf-8 -*-

__Date__ = '24 October 2019'
__author__ = ' Niraj Dev Pandey'
__Purpose__ = 'Coding Challenge for Brain One data scientist position'

import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.layers import Flatten
from keras.utils import plot_model
import keras.backend as K
from numpy import asarray
from numpy import zeros
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")


def library_check():
    """
    Check if user have the same python library as the developer machine
    :return: Raise attention if libraries version mismatched
    """
    if np.__version__ != '1.16.4':
        print("The project is developed on NumPy 1.16.4 ")
        print("you are running on numpy {} version".format(np.__version__))
    elif keras.__version__ != '2.3.1':
        print("The project is developed on keras 2.3.1")
        print("you are running on keras {} version".format(keras.__version__))
    elif pd.__version__ != '0.24.2':
        print("The project is developed on Pandas 0.24.2")
        print("you are running on Panda {} version".format(pd.__version__))
    elif sklearn.__version__ != '0.21.3':
        print("The project is developed on Sklearn 0.21.3 ")
        print("you are running on sklearn {} version".format(sklearn.__version__))
    else:
        print("congratulations...! you already have all the correct dependencies installed")


library_check()

data = []
for line in open('/home/niraj/Documents/Challenges/Brain_one/Clothing_Shoes_and_Jewelry_5.json', 'r'):
    data.append(json.loads(line))
# Dump json data to pandas DataFrame
df = pd.DataFrame(data)
df["rating"] = np.nan

# Modify rating column as per the ["overall"] column values
df.loc[df.overall == 5.0, 'rating'] = "+1"
df.loc[df.overall == 4.0, 'rating'] = "+1"
df.loc[df.overall == 1.0, 'rating'] = "-1"
df.loc[df.overall == 2.0, 'rating'] = "-1"
sns.factorplot(x='overall', data=df, kind='count', size=4, aspect=1.8)


def drop_useless_features(DataFrame, features):
    """
    Drop specified list of features which has no impact or less in our model building
    :param DataFrame: Complete DataFrame
    :param features: Useless features which you think that needs to be dropped before fitting the model.
    :return:
    """
    DataFrame.drop(features, inplace=True, axis=1)

    if features is None:
        raise FeatureNotProvided('Please provide the list of feature which you want to drop')
    return DataFrame


df = drop_useless_features(df, ["helpful",
                                "reviewTime",
                                "reviewerID",
                                "unixReviewTime",
                                "asin",
                                "reviewerName",
                                "overall",
                                "summary"])

# Drop Nan values as they are mostly the columns where the overall column has
# Neutral sentiment or 3.0. This is not our target anyway
df = df.dropna()

# Removing punctuations
# Converting to Lowercase and cleaning punctuations

df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join(text_to_word_sequence(x)))

# removing numbers from the column of reviewText

df['reviewText'] = df['reviewText'].str.replace('\d+', '')

# Plot positive and negative rating
plot_size = plt.rcParams["figure.figsize"]
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

df.rating.value_counts().plot(kind='pie', autopct='%1.0f%%')
df.reviewText.str.len().max()

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(df["reviewText"])
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(df["reviewText"])
# print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)

# load the whole embedding into memory
# This pre-trained embedding has been borrowed from Internet
embeddings_index = dict()
f = open('/content/drive/My Drive/Colab Notebooks/data/glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

X_train, X_test, y_train, y_test = train_test_split(
    padded_docs, df["rating"], test_size=0.2, random_state=1)


def precision(y_true, y_pred):
    """
    What proportion of positive identifications was actually correct?
    :param y_true: True label of out dependent variable
    :param y_pred: Predicted label by the model
    :return: precision accuracy score of our classification model
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    What proportion of actual positives was identified correctly?
    :param y_true: True label of out dependent variable
    :param y_pred: Predicted label by the model
    :return: recall accuracy score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# define model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall])

# summarize the model
print(model.summary())
# fit the model
model.fit(X_train, y_train, epochs=5, verbose=1)
# evaluate the model
plot_model(model, to_file='model.png')
evaluation = model.evaluate(X_test, y_test, verbose=1)

print("Loss: ", evaluation[0])
print("Accuracy: ", evaluation[1])
print("Precision: ", evaluation[2])
print("Recall: ", evaluation[3])
