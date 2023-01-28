from typing import Generator
import pandas as pd
import collections as col
import library.restructure as res
import library.const as con
import library.fileoperations as fo
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_aggregated_df(
    tweets_df_gen: Generator[pd.DataFrame, None, None],
    column_names: list
    ) -> pd.DataFrame:

    aggregated_df = pd.DataFrame()

    for tweets_df in tweets_df_gen:
        tmp_df = (
            tweets_df[column_names]
            .apply(pd.Series.value_counts)
            .rename_axis('len_values')
        )
        aggregated_df = pd.concat([aggregated_df, tmp_df])

    df = (
        aggregated_df
        .reset_index()
        .groupby('len_values')[column_names]
        .sum()
        .reset_index()
    )

    return df


def tweets_len_factor(
    tweets_proc_df_gen: Generator[pd.DataFrame, None, None],
    tweets_data_df_gen: Generator[pd.DataFrame, None, None]
    ) -> pd.DataFrame:

    tweets_len_factor_df = pd.DataFrame(
        columns={
            'id': np.uint64,
            'tweet_length': np.uint16,
            'counts_resp': np.uint32,
            'counts_enc': np.uint32,
            'resp_prob': np.float32,
            }
        )

    for tweets_proc_df, tweets_data_df in zip(tweets_proc_df_gen, tweets_data_df_gen):
        user_id = tweets_proc_df['author_id'].iloc[0]
        resp_ids, enc_ids = res.familiar_follower_tweets_ids_df(tweets_data_df)
        tmp_resp_df = (
            tweets_proc_df
            [tweets_proc_df['id'].isin(resp_ids)]
            .groupby('text_fixed_length')
            .size()
            .rename_axis('tweet_length')
            .reset_index(name='counts')
        )
        tmp_enc_df = (
            tweets_proc_df
            [tweets_proc_df['id'].isin(enc_ids)]
            .groupby('text_fixed_length')
            .size()
            .rename_axis('tweet_length')
            .reset_index(name='counts')
        )
        missing_resp_val = np.setdiff1d(np.arange(281), tmp_resp_df['tweet_length'].values)
        missing_enc_val = np.setdiff1d(np.arange(281), tmp_enc_df['tweet_length'].values)
        missing_resp_df = pd.DataFrame({'tweet_length': missing_resp_val, 'counts': np.zeros(missing_resp_val.shape[0])})
        missing_enc_df = pd.DataFrame({'tweet_length': missing_enc_val, 'counts': np.zeros(missing_enc_val.shape[0])})
        resp_df = pd.concat([tmp_resp_df, missing_resp_df], ignore_index=True)
        enc_df = pd.concat([tmp_enc_df, missing_enc_df], ignore_index=True)
        enc_df['id'] = user_id
        resp_df['id'] = user_id
        merged_df = pd.merge(resp_df, enc_df, on='tweet_length', suffixes=('_resp', '_enc'))
        merged_df['resp_prob'] = merged_df['counts_resp'] / merged_df['counts_enc']
        merged_df.drop(columns='id_enc', inplace=True)
        merged_df.rename(columns={'id_resp':'id'}, inplace=True)

        tweets_len_factor_df = pd.concat([tweets_len_factor_df, merged_df])


    tweets_len_factor_df = tweets_len_factor_df.query("resp_prob != 0")

    return tweets_len_factor_df


def average_by_user_tweets_len_factor(
        tweets_len_factor_df: pd.DataFrame
    ) -> pd.DataFrame:

    df = tweets_len_factor_df

    avg_by_user_df = (
        df
        .pivot_table(
            values='resp_prob',
            index=['tweet_length'],
            aggfunc='mean')
        .reset_index()
    )

    return avg_by_user_df
    

def average_by_all_tweets_len_factor(
        tweets_len_factor_df: pd.DataFrame
    ) -> pd.DataFrame:

    df = tweets_len_factor_df

    avg_by_all_df = (
        df
        .pivot_table(
            values=['counts_resp','counts_enc'],
            index=['tweet_length'],
            aggfunc='sum')
        .reset_index()
    )

    avg_by_all_df['resp_prob'] = avg_by_all_df['counts_resp'] / avg_by_all_df['counts_enc']

    return avg_by_all_df

    
def cosine_similarity_factor(
        tweets_proc_df_gen: Generator[pd.DataFrame, None, None],
        tweets_data_df_gen: Generator[pd.DataFrame, None, None],
        global_df_gen: Generator[pd.DataFrame, None, None],
        mode: str
    ):
    cos_sim_df = pd.DataFrame()

    vectors = get_global_idf(global_df_gen, mode)
    if mode == 'dev': cnt = 0

    for tweets_proc_df, (tweets_data_df, id) in zip(tweets_proc_df_gen, tweets_data_df_gen):
        A_id = id
        A_cos_sim_df = pd.DataFrame()
        resp_ids, enc_ids = res.familiar_follower_tweets_ids_df(tweets_data_df)
        proc_df = tweets_proc_df[['id','author_id','text_clean','type']]
        tweets_A_df = proc_df[proc_df['author_id'].isin([np.uint64(A_id)])].copy()
        tweets_B_df = proc_df[proc_df['id'].isin(enc_ids)].copy()
        tweets_A_df['text_clean'] = tweets_A_df['text_clean'].replace('\n', ' ', regex=True)
        tweets_B_df['text_clean'] = tweets_B_df['text_clean'].replace('\n', ' ', regex=True)

        doc_A = [' '.join(tweets_A_df["text_clean"].values)]

        for B_id, group_B in tweets_B_df.groupby('author_id'):
            doc_B = [' '.join(group_B["text_clean"].values)]

            corpus = np.concatenate((doc_A, doc_B))
            tfidf = vectors.transform(corpus)
            cos_sim = cosine_similarity(tfidf, tfidf)[0]
            resp_prob = group_B[group_B['id'].isin(resp_ids)].shape[0] / group_B.shape[0]
            tmp_A_cos_sim_df = pd.DataFrame({
                'author_A_id': A_id,
                'author_B_id': B_id,
                'cosine_similarity': cos_sim,
                'resp_prob': resp_prob
            })
            tmp_A_cos_sim_df = tmp_A_cos_sim_df.query("resp_prob != 0 and resp_prob < 0.99 and cosine_similarity < 0.99")
            A_cos_sim_df = pd.concat([A_cos_sim_df, tmp_A_cos_sim_df])
        cos_sim_df = pd.concat([cos_sim_df, A_cos_sim_df])

        if mode == 'dev':
            cnt += 1
            if cnt > 20: break

    return cos_sim_df


def get_global_idf(
        tweets_proc_df_gen: Generator[pd.DataFrame, None, None],
        mode: str
    ) -> TfidfVectorizer:

    tweets_A_df = pd.DataFrame()
    tweets_B_df = pd.DataFrame()

    if mode == 'dev': cnt = 0

    for tweets_proc_df in tweets_proc_df_gen:
        proc_df = tweets_proc_df[['id','author_id','text_clean']]
        user_A_id = proc_df['author_id'].iloc[0]

        tmp_A_df = proc_df[proc_df['author_id'].isin([user_A_id])]
        tmp_B_df = proc_df[~proc_df['author_id'].isin([user_A_id])]

        tweets_A_df = pd.concat([tweets_A_df, tmp_A_df])
        tweets_B_df = pd.concat([tweets_B_df, tmp_B_df])

        if mode == 'dev':
            cnt += 1
            if cnt > 20: break

    print(f'Tweets B df length before drop duplicates:  {len(tweets_B_df)}')
    tweets_B_df = tweets_B_df.drop_duplicates(subset='id')
    print(f'Tweets B df length after drop duplicates:  {len(tweets_B_df)}')

    tweets_A_df['text_clean'] = tweets_A_df['text_clean'].replace('\n', ' ', regex=True)
    tweets_B_df['text_clean'] = tweets_B_df['text_clean'].replace('\n', ' ', regex=True)


    vectors = TfidfVectorizer(min_df=1, stop_words="english")
    vectors.fit(np.concatenate((tweets_A_df['text_clean'].values, tweets_B_df['text_clean'].values)))

    return vectors


