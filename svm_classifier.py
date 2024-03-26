import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import re
import chardet
import matplotlib.pyplot as plt
import string
import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
import seaborn as sns
import collections

stop_words = set(stopwords.words('english')) - {'not'}

def preprocess_text(text):
    # remove RT, screen names, and URLs
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove stopwords
    text = ' '.join([word for word in text.lower().split() if word not in stop_words])

    return text

def top10_distribution(data):

    # filter data by label
    data_label_1 = data[data['Label'] == 1]
    data_label_2 = data[data['Label'] == 2]

    # initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

    # fit and transform text data into document-term matrix
    dtm_label_1 = tfidf_vectorizer.fit_transform(data_label_1['Tweet'])
    dtm_label_2 = tfidf_vectorizer.fit_transform(data_label_2['Tweet'])

    # sum up occurrence of each term across all documents for each label
    term_freq_label_1 = dtm_label_1.sum(axis=0)
    term_freq_label_2 = dtm_label_2.sum(axis=0)

    # get list of terms and their frequencies sorted in descending order for each label
    terms_label_1 = [(term, term_freq_label_1[0, i]) for term, i in tfidf_vectorizer.vocabulary_.items()]
    terms_label_2 = [(term, term_freq_label_2[0, i]) for term, i in tfidf_vectorizer.vocabulary_.items()]
    terms_label_1_sorted = sorted(terms_label_1, key=lambda x: x[1], reverse=True)[:10]
    terms_label_2_sorted = sorted(terms_label_2, key=lambda x: x[1], reverse=True)[:10]

    # print the top 10 most frequently occurring terms for each label
    print("Top 10 most frequent terms for Label 1:")
    for term, freq in terms_label_1_sorted:
        print(f"{term}: {freq}")

    print("\nTop 10 most frequent terms for Label 2:")
    for term, freq in terms_label_2_sorted:
        print(f"{term}: {freq}")

def character_stats(data):
    # Subset the data based on label
    label1_data = data[data['Label'] == 1]
    label2_data = data[data['Label'] == 2]

    # Calculate the length of texts in characters, excluding spaces, for each subset
    label1_lengths = label1_data['Tweet'].apply(lambda x: len(x.replace(' ', '')))
    label2_lengths = label2_data['Tweet'].apply(lambda x: len(x.replace(' ', '')))

    # Calculate the mean, standard deviation, minimum, and maximum length of texts for each subset
    label1_stats = [label1_lengths.mean(), label1_lengths.std(), label1_lengths.min(), label1_lengths.max()]
    label2_stats = [label2_lengths.mean(), label2_lengths.std(), label2_lengths.min(), label2_lengths.max()]

    # Print the statistics
    print(f"Label 1 statistics: mean={label1_stats[0]:.2f}, std.={label1_stats[1]:.2f}, min={label1_stats[2]}, max={label1_stats[3]}")
    print(f"Label 2 statistics: mean={label2_stats[0]:.2f}, std.={label2_stats[1]:.2f}, min={label2_stats[2]}, max={label2_stats[3]}")

def explicit_abuse_stats(test_data):
    results = {'Hashtag': [], 'Number of tweets with label 1': [], 'Number of tweets with explicit abuse': [], 'Percentage of label 1 tweets with explicit abuse': []}

    with open('hate_lexicon_wiegand.txt', 'r') as f:
        hate_lexicon = set([line.strip() for line in f])

    # Initialize dictionaries to store explicit abuse count and label 1 count for each hashtag
    hashtag_explicit_abuse_count = {}
    hashtag_label_1_count = {}

    waseem_hashtags = ['mkr', 'victimcard', 'FeminismIsCancer', 'sjw', 'WomenAgainstFeminism', 'islam terrorism', 'feminazi', 'immigrant', 'MKR', 'banislam', 'terrorism', 'Terrorism']
    for index, row in test_data.iterrows():
        tweet_hashtags = set(re.findall(r"#(\w+)", row['Tweet']))
        for hashtag in tweet_hashtags:
            if hashtag in waseem_hashtags:
                if hashtag not in hashtag_label_1_count:
                    hashtag_label_1_count[hashtag] = 0
                hashtag_label_1_count[hashtag] += 1

            tweet_words = set(row['Tweet'].split())
            if any(word in hate_lexicon for word in tweet_words):
                if hashtag not in hashtag_explicit_abuse_count:
                    hashtag_explicit_abuse_count[hashtag] = 0
                hashtag_explicit_abuse_count[hashtag] += 1

    # Calculate the percentage of label 1 tweets with explicit abuse for each hashtag
    for hashtag in hashtag_label_1_count.keys():
        if hashtag in hashtag_explicit_abuse_count:
            explicit_abuse_percentage = hashtag_explicit_abuse_count[hashtag] / hashtag_label_1_count[hashtag] * 100
        else:
            explicit_abuse_percentage = 0
 
        results['Hashtag'].append(hashtag)
        results['Number of tweets with label 1'].append(hashtag_label_1_count[hashtag])
        results['Number of tweets with explicit abuse'].append(hashtag_explicit_abuse_count.get(hashtag, 0))
        results['Percentage of label 1 tweets with explicit abuse'].append(explicit_abuse_percentage)

    # Convert the results to a pandas DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('explicit_abuse_stats_1.csv', index=False)

def hashtag_distribution(data):
    hashtags = []
    for tweet in data['Tweet']:
        hashtags += [i.lower() for i in tweet.split() if i.startswith('#')]
    counter = collections.Counter(hashtags)
    print('Hashtag distribution:')
    for hashtag, count in counter.most_common(10):
        print(f'{hashtag}: {count}')
    explicit_abuse = data[data['Label'] == 1].shape[0]
    total_tweets = data.shape[0]
    explicit_abuse_percentage = explicit_abuse/total_tweets * 100
    print(f"Percentage of explicit abuse in the Waseem test data: {explicit_abuse_percentage:.2f}%")

def top10_waseem_distribution(data):
    
    waseem_hashtags = ['mkr', 'victimcard', 'FeminismIsCancer', 'sjw', 'WomenAgainstFeminism', 
                       'islam terrorism', 'feminazi', 'immigrant', 'MKR', 'banislam', 'terrorism', 'Terrorism']

    # initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), 
                                       vocabulary=waseem_hashtags)

    # fit and transform text data into document-term matrix
    dtm = tfidf_vectorizer.fit_transform(data['Tweet'])

    # sum up occurrence of each term across all documents
    term_freq = dtm.sum(axis=0)

    # get list of terms and their frequencies sorted in descending order
    terms = [(term, term_freq[0, i]) for term, i in tfidf_vectorizer.vocabulary_.items()]
    terms_sorted = sorted(terms, key=lambda x: x[1], reverse=True)[:10]

    # print the top 10 most frequently occurring terms
    print("Top 10 most frequent waseem hashtags:")
    for term, freq in terms_sorted:
        print(f"{term}: {freq}")


with open('waseem/waseemtrain.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
train_data = pd.DataFrame({'text': lines})
train_data['text'] = train_data['text'].apply(preprocess_text)

train_labels = pd.read_csv('waseem/waseemtrainGold.txt', header=None, names=['target'])

# Define the pipeline for our SVM classifier
pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC())
])

# Define the grid of hyperparameters to search over
param_dist = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 3), (3, 4)],
    'vect__max_df': np.linspace(0.5, 1.0, num=6),
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf', 'sigmoid']
}

# Use random search to find the best hyperparameters
#random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=5, n_jobs=-1, verbose=1)
#random_search.fit(train_data['text'], train_labels['target'])

#pipeline.fit(train_data['text'], train_labels['target'])

#filename = 'hate_speech_classifier.sav'
#joblib.dump(random_search, filename)

pipeline = joblib.load('hate_speech_classifier_old.sav')

with open('waseem/waseemtest.txt', 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')
    
test_data = pd.DataFrame({'text': lines})
test_data['text'] = test_data['text'].apply(preprocess_text)
test_labels = pd.read_csv('waseem/waseemtestGold.txt', header=None, names=['target'])

predictions = pipeline.predict(test_data['text'])
print("Classification report for waseemtestGold: \n", classification_report(test_labels['target'], predictions))

# Detecting the encoding of the file
with open('hate_tweets-2.csv', 'rb') as f:
    result = chardet.detect(f.read())
    
# Reading the file with the detected encoding
test_data = pd.read_csv('hate_tweets-2.csv', encoding=result['encoding'])
test_data.dropna(subset=['Label'], inplace=True)
hashtag_distribution(test_data)
explicit_abuse_stats(test_data)
test_data['Tweet'] = test_data['Tweet'].apply(preprocess_text)
#top10_distribution(test_data)
top10_waseem_distribution(test_data)
character_stats(test_data)
predictions = pipeline.predict(test_data['Tweet'])
print("Classification report for tweets from last 2-3 months: \n", classification_report(test_data['Label'], predictions))