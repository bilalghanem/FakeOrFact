from os.path import join
import pandas as pd
import re
from features.LIWC.LIWC.liwc_readDict import readDict




class LIWC_NLI_class:

    def __init__(self, path=''):
        # LIWC, sad, anger, neg & pos emotion
        self.liwc = readDict(join(path, 'liwc.dic'))
        self.liwc = pd.DataFrame(self.liwc, columns=['word', 'category'])
        self.liwc['word'] = self.liwc['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.liwc['value'] = 1
        self.liwc = pd.pivot_table(self.liwc, index='word', columns=['category'],
                                   values='value', fill_value=0).reset_index().reindex(['word', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article',
                                                                                        'auxverb', 'past', 'present', 'future', 'adverb', 'preps',
                                                                                        'conj', 'quant', '', '', '',], axis=1)
        self.i = set(self.liwc[self.liwc['i'] == 1]['word'].tolist())
        self.we = set(self.liwc[self.liwc['we'] == 1]['word'].tolist())
        self.you = set(self.liwc[self.liwc['you'] == 1]['word'].tolist())
        self.shehe = set(self.liwc[self.liwc['shehe'] == 1]['word'].tolist())
        self.ipron = set(self.liwc[self.liwc['ipron'] == 1]['word'].tolist())
        self.article = set(self.liwc[self.liwc['article'] == 1]['word'].tolist())
        self.auxverb = set(self.liwc[self.liwc['auxverb'] == 1]['word'].tolist())
        self.past = set(self.liwc[self.liwc['past'] == 1]['word'].tolist())
        self.present = set(self.liwc[self.liwc['present'] == 1]['word'].tolist())
        self.future = set(self.liwc[self.liwc['future'] == 1]['word'].tolist())
        self.adverb = set(self.liwc[self.liwc['adverb'] == 1]['word'].tolist())
        self.preps = set(self.liwc[self.liwc['preps'] == 1]['word'].tolist())
        self.conj = set(self.liwc[self.liwc['conj'] == 1]['word'].tolist())
        self.quant = set(self.liwc[self.liwc['quant'] == 1]['word'].tolist())

    def score(self, sentence):
        words = sentence.split()
        i = len(set(words).intersection(self.i))
        we = len(set(words).intersection(self.we))
        you = len(set(words).intersection(self.you))
        shehe = len(set(words).intersection(self.shehe))
        ipron = len(set(words).intersection(self.ipron))
        article = len(set(words).intersection(self.article))
        auxverb = len(set(words).intersection(self.auxverb))
        past = len(set(words).intersection(self.past))
        present = len(set(words).intersection(self.present))
        future = len(set(words).intersection(self.future))
        adverb = len(set(words).intersection(self.adverb))
        preps = len(set(words).intersection(self.preps))
        conj = len(set(words).intersection(self.conj))
        quant = len(set(words).intersection(self.quant))

        return [i, we, you, shehe, ipron, article, auxverb, past, present, future, adverb, preps, conj, quant]

if __name__ == '__main__':
    sentence = "i'm we wondering, why there are not student here?"
    h = LIWC_NLI_class()
    print(h.score(sentence))
