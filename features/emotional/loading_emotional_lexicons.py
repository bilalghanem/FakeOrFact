import warnings, operator
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from os.path import join
from collections import OrderedDict


class emotional_lexicons:

    def __init__(self, path):
        self.lexicons_path = path

        # NRC, plutchik
        self.nrc = pd.read_csv(join(self.lexicons_path, 'nrc.txt'), sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()
        self.nrc.rename(columns={'negative': 'negative_emotion', 'positive': 'positive_emotion'}, inplace=True)
        del self.nrc['positive_emotion']
        del self.nrc['negative_emotion']

    def lex_NRC(self, sentence):
        splitted_sentence = sentence.split()
        result = [0, 0, 0, 0, 0, 0, 0, 0]
        for word in splitted_sentence:
            try:
                result = list(map(operator.add, result, self.nrc[self.nrc.word == str(word)].values.tolist()[0][1:]))
            except:
                pass
        return result

    def aggregated_vector_emo(self, sentence):
        emos = []
        splitted_sentence = str(sentence).split()
        for word in splitted_sentence:
            result_NRC = self.nrc[self.nrc.word == str(word)]
            if len(result_NRC) > 0:
                emos.append(result_NRC.values.tolist()[0][1:])
        x = np.sum(emos, 0).tolist() if len(emos) > 0 else [0, 0, 0, 0, 0, 0, 0, 0]
        return np.sum(emos, 0).tolist() if len(emos) > 0 else [0, 0, 0, 0, 0, 0, 0, 0]

