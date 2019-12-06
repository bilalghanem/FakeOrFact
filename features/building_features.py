import re, os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from ibm_watson import PersonalityInsightsV3
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os.path import join
from joblib import Parallel, delayed

from features.emotional.loading_emotional_lexicons import emotional_lexicons
from features.sentiment.loading_sentiment_analyzer import Sentiment
from features.LIWC.loading_LIWC import Lexical_LIWC

# config
np.random.seed(0)
tqdm.pandas()

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[[self.column]][self.column]

class emotional_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)

        file_name = './features/processed_data/emotional_features_{}.npy'.format(data_type)
        if exists(file_name):
            features = np.load(file_name)
        else:
            emo = emotional_lexicons(path=join(self.path, 'emotional'))
            loop = tqdm(X['text_cleaned'])
            loop.set_description('Building emotional_features')
            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(emo.aggregated_vector_emo)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return features

class sentiment_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)
        file_name = './features/processed_data/sentiment_features_{}.npy'.format(data_type)
        if exists(file_name):
            features = np.load(file_name)
        else:
            senti = Sentiment(path=join(self.path, 'sentiment'))
            loop = tqdm(X['text_cleaned'].tolist())
            loop.set_description('Building sentiment_features')
            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(senti.one_vector_senti)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return features

class LIWC_features(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)
        file_name = './features/processed_data/LIWC_features_{}.npy'.format(data_type)
        if exists(file_name):
            features = np.load(file_name)
        else:
            lex = Lexical_LIWC(path=join(self.path, 'LIWC'))
            loop = tqdm(X['text_cleaned'])
            loop.set_description('Building LIWC_features')
            features = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")(delayed(lex.one_vector_LIWC)(sentence) for sentence in loop)
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return features

class USEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, topic_based=False):
        self.path = path
        self.n_jobs = n_jobs
        self.topic_based = topic_based

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)
        file_name = './features/processed_data/USEncoder_{}.npy'.format(data_type)
        if exists(file_name):
            features = np.load(file_name)
        else:

            from models.USE import get_USE
            batch_size = 200
            arrow = 0
            features = []
            X = X['text_cleaned'].tolist()
            pbar = tqdm(total=(len(X)/batch_size))
            pbar.set_description('Building USEncoder_features')
            while arrow < len(X):
                if len(features) == 0:
                    features = get_USE(X[arrow:(arrow+batch_size)])
                    # print(arrow,' ',(arrow+batch_size))
                else:
                    features = np.vstack((features, get_USE(X[arrow:(arrow+batch_size)])))
                    # print(arrow, ' ', (arrow+batch_size))
                arrow += batch_size
                pbar.update(1)
            pbar.close()
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return features

class baselines(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1, bs_type='RAN'):
        self.path = path
        self.n_jobs = n_jobs
        self.bs_type = bs_type

    def fit(self, X, y=None):
        self.train_len = len(X)
        print('\nBaseline ({}):'.format(self.bs_type))
        x_train, y_train = self.shuffle_numpy(X['text'], X['label'])

        if self.bs_type == 'MC':
            self.mdl = Pipeline([
                ('vect', CountVectorizer(analyzer='word')),
                ('DC', DummyClassifier(strategy='most_frequent', random_state=100, constant=None))
            ])
        elif self.bs_type == 'RAN':
            self.mdl = Pipeline([
                ('vect', CountVectorizer(analyzer='word')),
                ('DC', DummyClassifier(strategy='uniform', random_state=100, constant=None))
            ])
        elif self.bs_type == 'BOW':
            self.mdl = Pipeline([
                ('vect', TfidfVectorizer(analyzer='word')),
                ('DC', LinearSVC(random_state=0)),
            ])
        self.mdl.fit(x_train, y_train)
        return self

    def shuffle_numpy(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def baselines(self, data):
        print(data.groupby('label').size())
        prediction = self.mdl.predict(data['text'])
        print('Acc:{:.2f}, Precision:{:.2f}, Recall:{:.2f}, F1_macro:{:.2f}, F1_micro:{:.2f}'.format(accuracy_score(data['label'], prediction),
                                                                    precision_score(data['label'], prediction, pos_label=0),
                                                                    recall_score(data['label'], prediction, pos_label=0),
                                                                    f1_score(data['label'], prediction, average='macro', pos_label=0),
                                                                    f1_score(data['label'], prediction, average='micro', pos_label=0)))
        # if self.bs_type == 'BOW':
        #     np.savetxt("SVM_BOW", prediction)
        exit(1)

    def transform(self, data):
        if self.train_len != len(data):
            self.baselines(data)
        return self

class BOW(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs
        self.TfidfVectorizer = TfidfVectorizer(analyzer='word', max_df=0.9, min_df=0.05)

    def fit(self, X, y=None):
        self.train_len = len(X)
        self.TfidfVectorizer.fit(X['text_cleaned'], y)
        print(len(self.TfidfVectorizer.get_feature_names()))
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)

        return self.TfidfVectorizer.transform(X['text_cleaned']).toarray()

class estepan_personality(BaseEstimator, TransformerMixin):

    def __init__(self, path='', n_jobs=1):
        self.path = path
        self.n_jobs = n_jobs
        self.indexer = {}
        self.load_vectors()

    def load_vectors(self):
        for type_ in ['train', 'dev', 'test']:
            with open(os.path.join(self.path, 'estepan_personality', 'users_{}.txt'.format(type_)), encoding='utf-8') as f:
                users = f.read().split('\n')[:-1]
                vectors = np.load(os.path.join(self.path, 'estepan_personality', 'personality_scores_{}.npy'.format(type_)))
                for i in range(len(users)):
                    if not users[i] in self.indexer:
                        self.indexer[users[i]] = vectors[i, :]

    def fit(self, X, y=None):
        self.train_len = len(X)
        return self

    def transform(self, X):
        if len(X) == self.train_len:
            data_type = 'train'
        elif len(X) == 393:
            data_type = 'dev'
        elif len(X) == 303:
            data_type = 'test'
        else:
            print('Error data length')
            exit(1)
        file_name = './features/processed_data/eduardo_personality_{}.npy'.format(data_type)
        if exists(file_name):
            features = np.load(file_name)
        else:
            features = []
            for user in X['user_name']:
                features.append(self.indexer[user])
            features = np.nan_to_num(np.array(features))
            np.save(file_name, features)
        return features


def manual_features(path='', n_jobs=1):
    manual_feats = Pipeline([
        ('FeatureUnion', FeatureUnion([
            # ('emotional_features', emotional_features(path=path, n_jobs=n_jobs)),
            # ('sentiment_features', sentiment_features(path=path, n_jobs=n_jobs)),
            ('LIWC_features', LIWC_features(path=path, n_jobs=n_jobs)),
            # ('USEncoder_features', USEncoder(path=path, n_jobs=n_jobs)),
            ('BOW', BOW(path=path, n_jobs=n_jobs)),
            ('estepan_personality', estepan_personality(path=path, n_jobs=n_jobs)),
        ], n_jobs=1)),
    ])
    return manual_feats


if __name__ == '__main__':
    df = pd.DataFrame([{'text': "I don't dfsf ih building"},
                       {'text': "I  want to be not rhgdf sad witt"}])
    res = manual_features(n_jobs=4).fit_transform(df)
    x = pd.Series(res.tolist())
    print('')
