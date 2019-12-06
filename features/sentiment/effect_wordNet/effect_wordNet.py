import pandas as pd
from builtins import range
from tqdm import tqdm
from os.path import exists
from nltk import WordNetLemmatizer


class effect_wordNet_class:
    def __init__(self, path=''):
        self.path = path
        if exists(str(path+'EffectWordNet_processed.tff')):
            self.neg = pd.read_csv(str(path+'EffectWordNet_processed.tff'), names=['id', 'polar', 'word', 'example'], sep='\t', header=None)
        else:
            self.neg = pd.read_csv(str(path+'EffectWordNet.tff'), names=['id', 'polar', 'word', 'example'], sep='\t', header=None)
            self.preprocess_data()
        self.positive = set(self.neg[self.neg['polar'] == '+Effect']['word'].tolist())
        self.negative = set(self.neg[self.neg['polar'] == '-Effect']['word'].tolist())
        self.WN_Lemma = WordNetLemmatizer()

    def preprocess_data(self):
        self.neg = self.neg[self.neg['polar'].isin(['+Effect', '-Effect'])]
        new = pd.DataFrame(columns=self.neg.columns.values)
        indexes = []
        for row in tqdm(self.neg['word'].copy()):
            res = self.neg[self.neg['word'] == row]
            words = res['word'].values[0].split(',')
            if len(words) > 1:
                for w in words:
                    res['word'] = w
                    new = new.append(res)
                indexes.append(res.index.values[0])

        self.neg.drop(self.neg.index[indexes], inplace=True)
        new = new.append(self.neg)
        self.neg = new.copy()
        self.neg.reset_index(inplace=True)
        new.to_csv(str(self.path+'EffectWordNet_processed.tff'), header=False, index=False, sep='\t')

    def score(self, sentence):
        words = sentence.split()
        for i in range(len(words)):
            words[i] = self.WN_Lemma.lemmatize(words[i])
        pos = len(set(words).intersection(self.positive))
        neg = len(set(words).intersection(self.negative))
        return [pos, neg]

if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    h = effect_wordNet_class()
    print(h.score('we are beautiifful but misfire catch'))
