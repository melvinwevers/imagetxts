import pandas as pd
import burst_detection as bd
import numpy as np
from collections import Counter
from itertools import dropwhile

'''
script to find bursts in text
'''

def read_df():
    df = pd.read_pickle('ads_image.pkl')
    # df = df.sample(frac=0.001)
    df['words'] = df['ocr_clean'].apply(lambda x: x.lower().split())
    df['publication_year'] = df['date'].apply(lambda x: x.year)
    df['publication_month'] = df['date'].apply(lambda x: x.month)
    df['publication_day'] = df['date'].apply(lambda x: x.day)
    #drop_cols = ['min_x', 'min_y', 'max_x', 'max_y', 'w', 'h',
    #             'ocr', 'image_url']
    #df.drop(drop_cols, axis=1, inplace=True)
    print(df.shape)
    return df


def find_unique_words(data, threshold):
    # count all words
    word_counts = Counter(data['words'].apply(pd.Series).stack())
    print('Number of unique words: ', len(word_counts))

    for key, count in dropwhile(lambda x: x[1] >= threshold,
                                word_counts.most_common()):
                                del word_counts[key]
    print('Number of unique words with at least', threshold, 'occurances: ',
          len(word_counts))
    # create a list of unique words
    return list(word_counts.keys())


def word_proportions(df, word_list):
    '''
    create dataframe with word word_proportions
    '''
    time_frame = ['publication_year', 'publication_month']
    d = df.groupby(['publication_year', 'publication_month'])['words'].count().reset_index(drop=True)
    all_r = pd.DataFrame(columns=word_list, index=d.index)
    for i, word in enumerate(word_list):
        all_r[word] = pd.concat([df.loc[:, time_frame],
                                df['words'].apply(lambda x: word in x)],
                                axis=1).groupby(by=time_frame) \
                               .sum().reset_index(drop=True)
        if np.mod(i, 100) == 0:
            print('total words', len(word_list), 'word', i, 'complete')
    return all_r, d


def find_bursts(d, all_r, word_list):
    '''
    burst detection function
    '''
    s = 2   # resolution of state jumps; higher s --> fewer but stronger bursts
    gam = 0.5  # difficulty of moving up a state; larger gamma --> harder to move up states, less bursty
    n = len(d)  # number of timepoints
    smooth_win = 5

    all_bursts = pd.DataFrame(columns=['begin', 'end', 'weight'])

    for i, word, in enumerate(word_list):
        r = all_r.loc[:, word].astype(int)

        # find the optimal state sequence (using the Viterbi algorithm)
        [q,d,r,p] = bd.burst_detection(r, d, n, s, gam, smooth_win)

        # enumerate the bursts
        bursts = bd.enumerate_bursts(q, word)

        # find weights of each burst
        bursts_weighted = bd.burst_weights(bursts, r, d, p)

        # add the weighted burst to list of all bursts
        all_bursts = all_bursts.append(bursts_weighted, ignore_index=True)

        # print a progress report every 100 words
        if np.mod(i, 100) == 0:
            print('total words', len(word_list), 'word', i, 'complete')

    return all_bursts.sort_values(by='weight', ascending=False)


def main():
    threshold = 10
    df = read_df()
    unique_words = find_unique_words(df, threshold)
    all_r, d = word_proportions(df, unique_words)
    all_r.to_pickle('all_unique_words.pkl')
    bursts_unique = find_bursts(d, all_r, unique_words)
    bursts_unique.to_pickle('burst_unique.pkl')


if __name__ == "__main__":
    main()
