from typing import Generator
import pandas as pd
import collections as col
import library.restructure as res
import library.const as con
import matplotlib.pyplot as plt

def tweets_text_len(tweets_df_gen: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
    aggregated_df = pd.DataFrame()

    for tweets_df in tweets_df_gen:
        tmp_df = (
            tweets_df['text_length']
            .value_counts()
            .rename_axis('len_values')
            .to_frame('len_counts')
        )
        aggregated_df = pd.concat([aggregated_df, tmp_df])


    df = (
        aggregated_df
        .reset_index()
        .groupby('len_values')['len_counts']
        .sum()
        .to_frame('len_counts')
        .reset_index()
    )

    return df


def tweets_text_len_2(tweets_df_gen: Generator[list, None, None]) -> pd.DataFrame:
    tweets_text_len_dict = col.defaultdict(int)

    for tweets_df in tweets_df_gen:
        tweets_text_df = tweets_df['text_length']

        for indx, tweet_text_len in tweets_text_df.items():
                tweets_text_len_dict[tweet_text_len] += 1        


    sorted_tweets_text_len_dict = col.OrderedDict(sorted(tweets_text_len_dict.items()))
    tweets_text_len_count_df = pd.DataFrame.from_dict(sorted_tweets_text_len_dict, orient='index')\
                                           .reset_index()
    tweets_text_len_count_df.columns = ['text_len','count']

    return tweets_text_len_count_df
