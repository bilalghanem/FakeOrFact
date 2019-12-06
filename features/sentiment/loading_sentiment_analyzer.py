import warnings, nltk
warnings.filterwarnings("ignore")
from os.path import join
from nltk.sentiment.util import mark_negation
from features.sentiment.effect_wordNet.effect_wordNet import effect_wordNet_class
from features.sentiment.subjectivity_clues.subjectivity_clues import subjectivity_clues_class
from features.sentiment.sentic_net.sentic_net import sentic_net_class
from features.sentiment.emo_lex.emo_lex import emo_lex_class

# nltk.download('wordnet')

class Sentiment:

    def __init__(self, path=''):
        # self.effect_WN = effect_wordNet_class(path=join(path, 'effect_wordNet/'))
        # self.sentic_net = sentic_net_class(path=join(path, 'sentic_net/'))
        # self.subj_cue_senti = subjectivity_clues_class(path=join(path, 'subjectivity_clues/'))
        self.emo_lex_senti = emo_lex_class(path=join(path, 'emo_lex/'))

    def one_vector_senti(self, sentence):
        # sentence = ' '.join(mark_negation(sentence.split()))
        sentence = str(sentence)
        global_vec = []
        # global_vec.extend(self.effect_WN.score(sentence))
        # global_vec.extend(self.sentic_net.score(sentence))
        # global_vec.extend(self.subj_cue_senti.score(sentence))
        global_vec.extend(self.emo_lex_senti.score(sentence))
        return global_vec

if __name__ == '__main__':
    snt = Sentiment()
    s = ["I like the movie, it's bad I hate it", "I love you sweety kiss", 'I was so sad, he called me the bitch, he was killed']
    print(snt.one_vector_senti(s[0]))
