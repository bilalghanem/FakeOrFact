import pandas as pd
import numpy as np
import json, pickle
from statsmodels.stats.contingency_tables import mcnemar

if __name__ == '__main__':
    data = pd.read_pickle('collecting/sent_anastasia/collected_users_concatenated.pickle')
    print('unique users: {}'.format(data.user.unique()))
    print()