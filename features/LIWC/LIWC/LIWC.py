from os.path import join
import pandas as pd
import re
from features.LIWC.LIWC.liwc_readDict import readDict




class LIWC_class:

    def __init__(self, path=''):
        # LIWC, sad, anger, neg & pos emotion
        self.liwc = readDict(join(path, 'liwc.dic'))
        self.liwc = pd.DataFrame(self.liwc, columns=['word', 'category'])
        self.liwc['word'] = self.liwc['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.liwc['value'] = 1
        categories = ['i', 'we', 'you', 'shehe', 'they', 'anx', 'past', 'present', 'future', 'work', 'leisure',
                      'home', 'money', 'relig', 'death', 'swear', 'assent', 'nonfl', 'filler', 'cause', 'discrep', 'tentat', 'certain']
        self.liwc = pd.pivot_table(self.liwc, index='word', columns=['category'],
                                   values='value', fill_value=0).reset_index().reindex(['word', *categories], axis=1)
        self.categories_sets = [set(self.liwc[self.liwc[item] == 1]['word'].tolist()) for item in categories]

    def score(self, sentence):
        words = sentence.split()
        results = [len(set(words).intersection(cat_set)) for cat_set in self.categories_sets]
        return results

if __name__ == '__main__':
    sentence = "i'm we wondering, why there are not student here?"
    h = LIWC_class()
    print(h.score(sentence))
