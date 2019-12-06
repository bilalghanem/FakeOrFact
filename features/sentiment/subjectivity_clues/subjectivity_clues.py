import re, warnings
import pandas as pd

warnings.filterwarnings("ignore")

class subjectivity_clues_class:
    def __init__(self, path=''):
        path = path + 'subjclueslen1-HLTEMNLP05.tff'
        self.SC = pd.read_csv(path, sep=' ', header=None, error_bad_lines=False, warn_bad_lines=False)
        del self.SC[0]
        del self.SC[1]
        del self.SC[3]
        del self.SC[4]

        self.SC = self.SC.applymap(lambda x: re.findall('(?<=[a-z0-9]=)\w+', x)[0])
        self.SC.columns = ['words', 'polar']
        print('')


    def score(self, sentence):
        words = sentence.split()
        pos = len(set(self.SC[(self.SC['words'].isin(words)) & (self.SC['polar'] == 'positive')]['words'].tolist()))
        neg = len(set(self.SC[(self.SC['words'].isin(words)) & (self.SC['polar'] == 'negative')]['words'].tolist()))
        return [pos, neg]

if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    h = subjectivity_clues_class()
    print(h.score('we are beautiful but mal'))
