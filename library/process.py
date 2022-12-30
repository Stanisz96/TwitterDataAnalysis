from typing import Generator
import pandas as pd
import collections as col
import library.restructure as res
import library.const as con

def tweets_text_len(tweets_df_gen: Generator[list, None, None]) -> pd.DataFrame:
    tweets_text_len_dict = col.defaultdict(int)

    for tweets_df in tweets_df_gen:
        # tweets_text_df = res.get_individual_tweets_text_len(tweets_df)
        tweets_text_df = res.get_individual_clean_tweets_text_df(tweets_df)
        break
        for indx, tweet_text_len in tweets_text_df.items():
                tweets_text_len_dict[tweet_text_len] += 1     


    sorted_tweets_text_len_dict = col.OrderedDict(sorted(tweets_text_len_dict.items()))
    tweets_text_len_count_df = pd.DataFrame.from_dict(sorted_tweets_text_len_dict, orient='index')\
                                           .reset_index()
    tweets_text_len_count_df.columns = ['text_len','count']

    return tweets_text_len_count_df


