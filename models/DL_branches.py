import re, inspect, time, os, pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from os.path import exists, join
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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
from keras.layers.normalization import BatchNormalization
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


class DL_branches:

    def __init__(self):
        # Settings
        self.scoring = 'f1'
        self.verbose = 1
        self.model_name = ''
        self.summary_table = {}
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(['a'])

        self.parameters = {'filter_sizes': (3, 5),
                           'num_filters': 8,
                           'Dense1_size': 64,
                           'Dense2_size': 32,
                           'rnn_size': 32,
                           'Dropout': 0.53364,
                           'Activation': 'selu',
                           'Optimizer': 'adadelta',
                           'MAX_SEQUENCE_LENGTH': 2000,
                           'EmbeddingSize': 300,
                           'EMBEDDING_PATH': 'D:/glove/glove.840B.300d.txt',
                           'MaxEpoch': 15,
                           'BatchSize': 4,
                           'vocab': 0
                           }

    def prepare_input_branches(self, data, feats_train, feats_dev, feats_test):
        self.read_Data_branches(data, feats_train, feats_dev, feats_test)
        self.prep_text()
        self.prep_embed()

    def read_Data_branches(self, data, feats_train, feats_dev, feats_test):
        train, dev, test = data.train, data.dev, data.test

        train['text_cleaned'] = train['text_cleaned'].map(lambda text: text.replace('.', ''))
        dev['text_cleaned'] = dev['text_cleaned'].map(lambda text: text.replace('.', ''))
        test['text_cleaned'] = test['text_cleaned'].map(lambda text: text.replace('.', ''))

        train = train[['text_cleaned', 'label']]
        train.rename(columns={'text_cleaned': 'text'}, inplace=True)
        dev = dev[['text_cleaned', 'label']]
        dev.rename(columns={'text_cleaned': 'text'}, inplace=True)
        test = test[['text_cleaned', 'label']]
        test.rename(columns={'text_cleaned': 'text'}, inplace=True)

        self.train = {col: train[col].tolist() for indx, col in enumerate(train)}
        self.train['features'] = feats_train
        self.dev = {col: dev[col].tolist() for indx, col in enumerate(dev)}
        self.dev['features'] = feats_dev
        self.test = {col: test[col].tolist() for indx, col in enumerate(test)}
        self.test['features'] = feats_test

    def prep_text(self):
        self.indexer = Tokenizer(lower=True, num_words=100000)
        self.indexer.fit_on_texts(self.train['text'] + self.dev['text'] + self.test['text'])
        self.parameters['vocab'] = len(self.indexer.word_counts) + 1

        # 3, Convert each word in sent to num and zero pad
        def padding(x, MaxLen):
            return pad_sequences(sequences=self.indexer.texts_to_sequences(x), maxlen=MaxLen)
        def pad_data(x):
            if 'label' in x:
                return {'text': padding(x['text'], self.parameters['MAX_SEQUENCE_LENGTH']), 'label': self.preparing_labels(x['label']), 'features': x['features']}
            else:
                return {'text': padding(x['text'], self.parameters['MAX_SEQUENCE_LENGTH']), 'features': x['features']}

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
        x = exists(join(path, '{}_branches.npy'.format(model_name)))
        if x:
            for filename in os.listdir(path):
                if filename == '{}_branches.npy'.format(model_name):
                    embedding_matrix = np.load(join(path, filename))
                    break
        else:
            embedding_matrix = self.load_embeddings()
            np.save(join(path, '{}_branches.npy'.format(model_name)), embedding_matrix)

        self.Embed = Embedding(input_dim=self.parameters['vocab'],
                               output_dim=self.parameters['EmbeddingSize'],
                               input_length=self.parameters['MAX_SEQUENCE_LENGTH'],
                               trainable=True,
                               weights=[embedding_matrix],
                               name='Embed_Layer')

    def Network_branches_CNN(self):
        self.model_name = inspect.currentframe().f_code.co_name

        inp_branch1 = Input(shape=(self.train['features'].shape[1],), name='branch1_input')
        x = Dense(self.parameters['Dense1_size'], activation=self.parameters['Activation'], name='branch1_Dense')(inp_branch1)
        branch1 = Model(inp_branch1, x, name='branch1_')

        inp_branch2 = Input(shape=(self.parameters['MAX_SEQUENCE_LENGTH'],), name='branch2_input')
        z = self.Embed(inp_branch2)
        z = Dropout(self.parameters['Dropout'])(z)
        conv_blocks = []
        for sz in self.parameters['filter_sizes']:
            conv = Convolution1D(filters=self.parameters['num_filters'], kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dropout(self.parameters['Dropout'])(z)
        branch2 = Model(inp_branch2, z, name='branch2_')

        merged = concatenate([branch1.output, branch2.output])
        merged = Dense(self.parameters['Dense2_size'], activation=self.parameters['Activation'])(merged)
        merged = Dense(2, activation="softmax")(merged)
        self.model = Model(inputs=[branch1.input, branch2.input], outputs=merged)

    def Network_branches_LSTM(self):
        self.model_name = inspect.currentframe().f_code.co_name

        inp_branch1 = Input(shape=(self.train['features'].shape[1],), name='branch1_input')
        x = Dense(self.parameters['Dense1_size'], activation=self.parameters['Activation'], name='branch1_Dense')(inp_branch1)
        branch1 = Model(inp_branch1, x, name='branch1_')

        inp_branch2 = Input(shape=(self.parameters['MAX_SEQUENCE_LENGTH'],), name='branch2_input')
        z = self.Embed(inp_branch2)
        z = CuDNNLSTM(self.parameters['rnn_size'])(z)
        z = Dropout(self.parameters['Dropout'])(z)
        branch2 = Model(inp_branch2, z, name='branch2_')

        merged = concatenate([branch1.output, branch2.output])
        merged = Dense(self.parameters['Dense2_size'], activation=self.parameters['Activation'])(merged)
        merged = Dense(2, activation="softmax")(merged)
        self.model = Model(inputs=[branch1.input, branch2.input], outputs=merged)


    def evaluate_on_test(self):
        loss, acc = self.model.evaluate([self.test['features'], self.test['text']], self.test['label'], batch_size=self.parameters['BatchSize'], verbose=self.verbose)
        print("@@ Test: loss = {:.5f}, acc = {:.3f}%".format(loss, acc))
        Y_test_pred = self.model.predict([self.test['features'], self.test['text']], batch_size=self.parameters['BatchSize'], verbose=0)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(self.test['label'], axis=1)
        print('acc:{:.2f}, precision:{:.2f}, recall:{:.2f}, ma-F1:{:.2f}, mi-F1:{:.2f}'.format(accuracy_score(Y_test, Y_test_pred),
                                                                                               precision_score(Y_test, Y_test_pred),
                                                                                               recall_score(Y_test, Y_test_pred),
                                                                                               f1_score(Y_test, Y_test_pred, average='macro'),
                                                                                               f1_score(Y_test, Y_test_pred, average='micro')))
        np.savetxt('{}+LIEMOSENTPERS'.format(self.__class__.__name__), Y_test_pred)

    def run_model(self, reTrain=True):
        # 1, Compile the model
        self.model.compile(optimizer=self.parameters['Optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

        # 2, Prep
        callback = [EarlyStopping(min_delta=0.1, patience=4, mode='min'),
                    # ReduceLROnPlateau(patience=2, verbose=2),
                    ModelCheckpoint('./models/preprocessed_data/{}.check'.format(self.model_name), save_best_only=False, save_weights_only=False)]
        # 3, Train
        if reTrain:
            self.model.fit(x=[self.train['features'], self.train['text']], y=self.train['label'], batch_size=self.parameters['BatchSize'], epochs=self.parameters['MaxEpoch'], verbose=self.verbose,
                           validation_data=([self.dev['features'], self.dev['text']], self.dev['label']), callbacks=callback)
        else:
            fn = './models/preprocessed_data/{}_branches.check'.format(self.model_name)
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
                        'dense1': hp.choice('dense1_size', [8, 16, 32, 64]),
                        'dense2': hp.choice('dense2_size', [8, 16, 32, 64]),
                        'filter_sizes': hp.choice('Filter_Sizes', [(3,), (4,), (5,), (6,), (2,3), (2,3,4), (3,4), (3,4,5), (2,4), (3,5)]),
                        'num_filters': hp.choice('Num_Filters', [8, 16, 32, 50, 64]),
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

        if type(self.summary_table['score']) != list:
            df_summary_table = pd.DataFrame(self.summary_table)#, index=[0])
        else:
            df_summary_table = pd.DataFrame(self.summary_table)
        df_summary_table.sort_values('score', inplace=True, ascending=False)
        df_summary_table.to_csv('./models/output/results_branches.csv', header=True, index=False)

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
        self.parameters['Dense1_size'] = params['dense1']
        self.parameters['Dense2_size'] = params['dense2']
        self.parameters['filter_sizes'] = params['filter_sizes']
        self.parameters['num_filters'] = params['num_filters']

        print('Current: {}'.format(params))
        self.Network_branches_CNN()
        self.run_model()
        Y_dev_pred = self.model.predict([self.test['features'], self.test['text']], batch_size=self.parameters['BatchSize'], verbose=0)
        Y_dev_pred = np.argmax(Y_dev_pred, axis=1)
        self.Y_dev = np.argmax(self.test['label'], axis=1)

        if self.scoring.lower() == 'f1':
            return f1_score(self.Y_dev, Y_dev_pred, average='macro')
        elif self.scoring.lower() == 'acc':
            return accuracy_score(self.Y_dev, Y_dev_pred)