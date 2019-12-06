import os, json
import pandas as pd
from tqdm import tqdm


def append_to_dict(tweet, user):
    tweet['user_name'] = user
    return tweet

def read_anas_format(top=1000):

    if os.path.isfile('users') and os.path.isfile('tweets'):
        users = pd.read_pickle('users')
        tweets = pd.read_pickle('tweets')
    else:
        file = 'Results_leaderStories_userLists.csv'
        users = pd.read_csv(file, header=0)
        users = users.rename(columns={'user': 'user_name', 'annotation': 'label'})
        tweets = []
        for item in tqdm(os.listdir('./all_users')):
            user = open('./all_users/{}'.format(item), 'r', encoding='utf-8')
            data = json.load(user)[:top]
            data = [append_to_dict(tweet, item) for tweet in data]
            tweets.extend(data)
        tweets = pd.DataFrame(tweets)

        users = users[users['user_name'].isin(tweets['user_name'])]
        users.to_pickle('users')
        tweets.to_pickle('tweets')
    print()


if __name__ == '__main__':
    read_anas_format()