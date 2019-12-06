from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import pickle
from tqdm import tqdm
from features.building_features import manual_features
from features.building_features import baselines
import numpy as np

# config
np.random.seed(0)


class Model:

    def __init__(self, scoring='accuracy',  verbose=0):
        self.scoring = scoring
        self.verbose = verbose
        self.path = './features'

    def run_model(self, data):

        feats = Pipeline([
            ('main_pip', Pipeline([
                ('baselines', baselines(path=self.path, n_jobs=-1, bs_type='BOW')),
                # ('manual_features', manual_features(path=self.path, n_jobs=-1)),
                # ('Normalization', StandardScaler()),
            ])),
        ])

        x_train, y_train = feats.fit_transform(data.train, data.train['label']), data.train['label']
        x_test, y_test = feats.transform(data.test), data.test['label']

        model = LogisticRegression(n_jobs=-1, random_state=0)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        print('Acc:{:.2f}, Precision:{:.2f}, Recall:{:.2f}, F1_macro:{:.2f}, F1_micro:{:.2f}'.format(
            accuracy_score(y_test, prediction),
            precision_score(y_test, prediction, pos_label=0),
            recall_score(y_test, prediction, pos_label=0),
            f1_score(y_test, prediction, average='macro', pos_label=0),
            f1_score(y_test, prediction, average='micro', pos_label=0)))
        # np.savetxt("USE", prediction)
        exit(1)