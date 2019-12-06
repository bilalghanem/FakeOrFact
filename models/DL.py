import re, inspect, time, os, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from os.path import exists, join
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import Parallel, delayed
import warnings, random
from models.utils import f1_score as f1_

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Input, Dropout, concatenate, Bidirectional, CuDNNLSTM, Conv1D, MaxPooling1D, Convolution1D, Concatenate, Flatten, MaxPooling1D, TimeDistributed
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, CSVLogger
from keras.utils import to_categorical
from hyperopt import fmin, tpe, hp, Trials
from hyperopt import STATUS_OK

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Reproducibility
import tensorflow as tf
np.random.seed(0)
random.seed(0)
tf.set_random_seed(0)
tf.random.set_random_seed(0)

warnings.filterwarnings('ignore')


class DL:

    def __init__(self):
        # Settings
        self.scoring = 'f1'
        self.verbose = 1
        self.model_name = ''
        self.summary_table = {}
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(['a'])

        self.parameters = {'lstm_size': 64,
                           'Dense_size': 8,
                           'Dropout': 0.4895,
                           'Activation': 'tanh',
                           'Optimizer': 'rmsprop',
                           'MAX_SEQUENCE_LENGTH': 3000,
                           'EmbeddingSize': 300,
                           'EMBEDDING_PATH': 'D:/glove/glove.840B.300d.txt',
                           'MaxEpoch': 7,
                           'BatchSize': 4,
                           'vocab': 0
                           }

    def prepare_input(self, train, dev, test):
        self.read_Data(train, dev, test)
        self.prep_text()
        self.prep_embed()

    def read_Data(self, train, dev, test):
        train = train[['text_cleaned', 'label']].rename(columns={'text_cleaned': 'text'})
        dev = dev[['text_cleaned', 'label']].rename(columns={'text_cleaned': 'text'})
        test = test[['text_cleaned', 'label']].rename(columns={'text_cleaned': 'text'})

        self.train = {col: train[col].tolist() for indx, col in enumerate(train)}
        self.dev = {col: dev[col].tolist() for indx, col in enumerate(dev)}
        self.test = {col: test[col].tolist() for indx, col in enumerate(test)}

    def prep_text(self):
        self.indexer = Tokenizer(lower=True, num_words=100000)
        self.indexer.fit_on_texts(self.train['text'] + self.dev['text'] + self.test['text'])
        self.parameters['vocab'] = len(self.indexer.word_counts) + 1

        # 3, Convert each word in sent to num and zero pad
        def padding(x, MaxLen):
            return pad_sequences(sequences=self.indexer.texts_to_sequences(x), maxlen=MaxLen)
        def pad_data(x):
            if 'label' in x:
                return {'text': padding(x['text'], self.parameters['MAX_SEQUENCE_LENGTH']), 'label': self.preparing_labels(x['label'])}
            else:
                return {'text': padding(x['text'], self.parameters['MAX_SEQUENCE_LENGTH'])}

        self.train = pad_data(self.train)
        self.dev = pad_data(self.dev)
        self.test = pad_data(self.test)

    def preparing_labels(self, y):
        y = np.array(y)
        y = y.astype(str)
        if y.dtype.type == np.array(['a']).dtype.type:
            if len(self.labelencoder.classes_) < 2:
                self.labelencoder.fit(y)
                self.Labels = self.labelencoder.classes_.tolist()
            y = self.labelencoder.transform(y) #labelencoder.inverse_transform(y)
        # https://github.com/keras-team/keras/issues/3109
        labels = to_categorical(y, len(self.Labels))
        return labels

    def load_embeddings(self):
        # Creat a embedding matrix for word2vec(use GloVe)
        embed_index = {}
        model_name = self.parameters['EMBEDDING_PATH'][self.parameters['EMBEDDING_PATH'].rfind('/')+1:]
        LEM_loop = tqdm(open(self.parameters['EMBEDDING_PATH'], 'r', encoding='utf-8'))
        LEM_loop.set_description('Loading {} model'.format(model_name))
        def loading(line):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            return word, coefs
        embed_index[0] = dict(Parallel(-1)(delayed(loading)(line) for line in LEM_loop))
        LEM_loop.close()


        embedding_matrix = np.zeros((self.parameters['vocab'], self.parameters['EmbeddingSize']))
        unregistered = []
        for ei in embed_index:
            EM_loop = tqdm(self.indexer.word_index.items())
            EM_loop.set_description('Loading vocab words of EM({})'.format(ei))
            for word, i in EM_loop:
                if word in embed_index[ei]:
                    embedding_matrix[i] = embed_index[ei][word]
                else:
                    unregistered.append(word)
        unregistered = list(set(unregistered))
        open('./models/preprocessed_data/unregisterd_word.txt'
             , 'w', encoding="utf-8").write(str(unregistered))
        return embedding_matrix

    def prep_embed(self):
        path = './models/preprocessed_data'
        model_name = self.parameters['EMBEDDING_PATH'][self.parameters['EMBEDDING_PATH'].rfind('/') + 1:]
        x = exists(join(path, '{}.npy'.format(model_name)))
        if x:
            for filename in os.listdir(path):
                if filename == '{}.npy'.format(model_name):
                    embedding_matrix = np.load(join(path, filename))
                    break
        else:
            embedding_matrix = self.load_embeddings()
            np.save(join(path, '{}.npy'.format(model_name)), embedding_matrix)

        self.Embed = Embedding(input_dim=self.parameters['vocab'],
                               output_dim=self.parameters['EmbeddingSize'],
                               input_length=self.parameters['MAX_SEQUENCE_LENGTH'],
                               trainable=True,
                               weights=[embedding_matrix],
                               name='Embed_Layer')

    def Network(self):
        self.model_name = inspect.currentframe().f_code.co_name
        inp = Input(shape=(self.parameters['MAX_SEQUENCE_LENGTH'],))
        z = self.Embed(inp)
        z = CuDNNLSTM(self.parameters['lstm_size'])(z)
        z = Dropout(self.parameters['Dropout'])(z)
        out = Dense(2, activation="softmax")(z)
        self.model = Model(inputs=inp, outputs=out)


    def evaluate_on_test(self):
        loss, acc = self.model.evaluate(self.test['text'], self.test['label'], batch_size=self.parameters['BatchSize'], verbose=self.verbose)
        print("@@ Test: loss = {:.5f}, acc = {:.3f}%".format(loss, acc))
        Y_test_pred = self.model.predict(self.test['text'], batch_size=self.parameters['BatchSize'], verbose=0)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(self.test['label'], axis=1)
        print('acc:{:.2f}, precision:{:.2f}, recall:{:.2f}, ma-F1:{:.2f}, mi-F1:{:.2f}'.format(accuracy_score(Y_test, Y_test_pred), precision_score(Y_test, Y_test_pred), recall_score(Y_test, Y_test_pred),
                                                                                               f1_score(Y_test, Y_test_pred, average='macro'),
                                                                                               f1_score(Y_test, Y_test_pred, average='micro')))

    def run_model(self, reTrain=True):
        # 1, Compile the model
        self.model.compile(optimizer=self.parameters['Optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

        # 2, Prep
        callback = [EarlyStopping(min_delta=0.003, patience=4, restore_best_weights=True),
                    # ReduceLROnPlateau(patience=2, verbose=2),
                    ModelCheckpoint('./models/preprocessed_data/{}.check'.format(self.model_name), save_best_only=False, save_weights_only=False)]
        # 3, Train
        if reTrain:
            self.model.fit(x=self.train['text'], y=self.train['label'], batch_size=self.parameters['BatchSize'], epochs=self.parameters['MaxEpoch'], verbose=self.verbose,
                           validation_data=(self.dev['text'], self.dev['label']), callbacks=callback)
        else:
            fn = './models/preprocessed_data/{}.check'.format(self.model_name)
            if os.path.exists(fn):
                self.model.load_weights(fn, by_name=True)
                print('--------Load Weights Successful!--------')

        # 4, Evaluate
        if reTrain:
            self.evaluate_on_test()


    def run_hyperopt_search(self, n_evals):
        self.verbose = 0
        self.pbar = tqdm(total=n_evals)
        self.pbar.set_description('Hyperopt evals')
        search_space = {'optimizer': hp.choice('opt', ['adadelta', 'adam', 'rmsprop', 'sgd']),
                        'activation': hp.choice('act', ['selu', 'relu', 'tanh', 'elu']),
                        'drop': hp.uniform('drop_1', 0.1, 0.7),
                        'lstm_size': hp.choice('lstm_size', [8, 16, 32, 64]),
                        }
        trials = Trials()
        best = fmin(self.objective_function, space=search_space, algo=tpe.suggest, max_evals=n_evals, trials=trials)
        self.pbar.close()
        bp = trials.best_trial['result']['Params']
        print('\n\n', best)
        print(bp)

    def objective_function(self, params):
        mean_score = self.Kstratified(params)
        params.update({'score': mean_score})
        print(params)

        if len(self.summary_table) < 1:
            self.summary_table.update(params)
        else:
            for key, value in self.summary_table.items():
                if key in params:
                    values = self.summary_table[key]
                    if not type(values) is list:
                        values = [values]
                    values.append(params[key])
                    self.summary_table[key] = values

        try:
            df_summary_table = pd.DataFrame(self.summary_table, index=[0])
        except:
            df_summary_table = pd.DataFrame(self.summary_table)
        df_summary_table.sort_values('score', inplace=True, ascending=False)
        df_summary_table.to_csv('./models/output/results.csv', header=True, index=False)

        output = {'loss': 1 - mean_score,
                  'Params': params,
                  'status': STATUS_OK,
                  }
        self.pbar.update(1)
        return output

    def Kstratified(self, params):
        self.parameters['Dropout'] = params['drop']
        self.parameters['Activation'] = params['activation']
        self.parameters['Optimizer'] = params['optimizer']
        self.parameters['lstm_size'] = params['lstm_size']

        print('Current: {}'.format(params))
        self.Network()
        self.run_model()
        Y_dev_pred = self.model.predict(self.dev['text'], batch_size=self.parameters['BatchSize'], verbose=0)
        Y_dev_pred = np.argmax(Y_dev_pred, axis=1)
        self.Y_dev = np.argmax(self.dev['label'], axis=1)

        if self.scoring.lower() == 'f1':
            return f1_score(self.Y_dev, Y_dev_pred, average='macro')
        elif self.scoring.lower() == 'acc':
            return accuracy_score(self.Y_dev, Y_dev_pred)

    def bert(self, data):
        from models.KB import main as train_Bert
        self.prepare_bert(data)
        train_Bert(self.train, self.dev, self.test)

    def prepare_bert(self, data):
        data.rename(columns={'all_tweets': 'text'}, inplace=True)
        data['text'].replace('', 'test', inplace=True)
        data['len'] = data['text'].map(lambda text: len(text.split()))
        data = data.sort_values('len', ascending=True)
        data = data[['text', 'label']]

        msk_dev = np.random.rand(len(data)) < 0.25
        self.org_dev = data[msk_dev]
        data = data[~msk_dev]

        msk_test = np.random.rand(len(data)) < 0.15
        self.org_test = data[msk_test]
        self.org_train = data[~msk_test]

        self.org_train.reset_index(inplace=True, drop=True)
        self.org_dev.reset_index(inplace=True, drop=True)
        self.org_test.reset_index(inplace=True, drop=True)

        self.train = self.org_train
        self.dev = self.org_dev
        self.test = self.org_test

        self.train = {col: self.train[col].tolist() for indx, col in enumerate(self.train)}
        self.dev = {col: self.dev[col].tolist() for indx, col in enumerate(self.dev)}
        self.test = {col: self.test[col].tolist() for indx, col in enumerate(self.test)}