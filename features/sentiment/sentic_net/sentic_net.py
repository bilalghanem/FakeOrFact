import pandas as pd
from os.path import exists

class sentic_net_class:

    def __init__(self, path=''):
        path = path + 'senticnet5.csv'

        if not exists(path):
            f = open('senticnet5.py', mode='r')
            code = f.read()
            exec(code, globals())
            self.SN = pd.DataFrame(senticnet)
            self.SN = self.SN.transpose()
            self.SN.reset_index(level=[0], inplace=True)
            self.SN.columns = ['words', 'pleasantness_value', 'attention_value', 'sensitivity_value', 'aptitude_value', 'primary_mood', 'secondary_mood', 'polarity_label', 'polarity_value', 'semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']
            self.SN.to_csv(path, header=True, index=False)
        else:
            self.SN = pd.read_csv(path, header=0, index_col=None)
        print('')

    def score(self, sentence):
        words = sentence.split()
        pos = sum(self.SN[((self.SN['words'].isin(words)) & (self.SN['polarity_label'] == 'positive'))]['polarity_value'].tolist())
        neg = sum(self.SN[((self.SN['words'].isin(words)) & (self.SN['polarity_label'] == 'negative'))]['polarity_value'].tolist())
        return [pos, neg]

if __name__ == '__main__':
    s = ['love', 'hate', 'shit', 'hit']
    h = sentic_net_class()
    print(h.score('we are beautiful but mal'))
