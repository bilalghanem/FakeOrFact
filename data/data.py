# -*- coding: utf-8 -*-
import os, re, ast, pickle, json, time
from datetime import datetime
from tqdm import tqdm
from os.path import join
from joblib import delayed, Parallel
from dateutil import parser
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


tqdm.pandas()


class data:

    def __init__(self):
        self.path = './data/'
        self.users = []
        self.tweets = []

    def clean_data(self, text):
        #-URL
        text = re.sub(r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})", " ", text)
        #-USER_MENTION
        text = re.sub(r'\@(\w+)', ' ', text)
        #-HASH_TAG
        text = re.sub(r'\#(\w+)', ' ', text)
        #-\n
        text = re.sub(r'\n', ' ', text)
        #-\r
        text = re.sub(r'\r', ' ', text)

        text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove uneeded special characters
        text = re.sub(r'rt', '', text)

        text = re.sub(r'\s{2,}', ' ', text)  # remove extra spaces
        return text.strip().lower()

    def split_Data(self):
        msk_dev = np.random.rand(len(self.users)) < 0.20
        self.dev = self.users[msk_dev]
        self.users = self.users[~msk_dev]
        msk_test = np.random.rand(len(self.users)) < 0.20
        self.test = self.users[msk_test]
        self.train = self.users[~msk_test]
        self.train = self.train.reset_index(drop=True)
        self.dev = self.dev.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

    def load_data(self, top=1000):
        start = time.time()
        if os.path.isfile(join(self.path, 'train')) and os.path.isfile(join(self.path, 'dev')) and os.path.isfile(join(self.path, 'test')):
            self.train = pd.read_pickle(join(self.path, 'train'))
            self.dev = pd.read_pickle(join(self.path, 'dev'))
            self.test = pd.read_pickle(join(self.path, 'test'))
            print('========= All the data loaded =========')
        else:
            file = join(self.path, 'Results_leaderStories_userLists.csv')
            self.users = pd.read_csv(file, header=0)
            self.users = self.users.rename(columns={'user': 'user_name', 'annotation': 'label'})

            def append_to_dict(tweet, user):
                tweet['user_name'] = user
                return tweet

            tweets = []
            for item in tqdm(os.listdir(join(self.path, 'all_users'))):
                user = open(join(self.path, 'all_users', item), 'r', encoding='utf-8')
                data = json.load(user)[:top]
                data = [append_to_dict(tweet, item) for tweet in data]
                tweets.extend(data)
            self.tweets = pd.DataFrame(tweets)
            self.users = self.users[self.users['user_name'].isin(self.tweets['user_name'].tolist())]
            self.tweets.drop_duplicates(subset=['text'], keep='first', inplace=True)
            self.users.drop_duplicates(subset=['user_name'], keep='first', inplace=True)
            self.tweets = self.tweets[['user_name', 'text', 'created_at', 'id_str']]
            self.tweets['text_cleaned'] = self.tweets['text'].progress_map(lambda text: self.clean_data(text))

            """Sort users file to fix it later"""
            self.users = shuffle(self.users)
            self.users['text'] = self.users['user_name'].progress_map(lambda user_name: '. '.join(self.tweets[self.tweets['user_name'] == user_name]['text'].tolist()))
            self.users['text_cleaned'] = self.users['user_name'].progress_map(lambda user_name: '. '.join(self.tweets[self.tweets['user_name'] == user_name]['text_cleaned'].tolist()))
            self.users.reset_index(drop=True, inplace=True)

            self.split_Data()

            self.train.to_pickle(join(self.path, 'train'))
            self.dev.to_pickle(join(self.path, 'dev'))
            self.test.to_pickle(join(self.path, 'test'))
        print('%0.2f seconds.\n' % (time.time() - start))