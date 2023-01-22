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


def cosine_similarity_factor_gen(
        tweets_proc_df_gen: Generator[pd.DataFrame, None, None],
        tweets_data_df_gen: Generator[pd.DataFrame, None, None],
        compare_by_tweet: bool
    ) -> Generator[tuple[pd.DataFrame, np.uint64], None, None]:

    for tweets_proc_df, tweets_data_df in zip(tweets_proc_df_gen, tweets_data_df_gen):
        user_id = tweets_proc_df['author_id'].iloc[0]
        resp_ids, enc_ids = res.familiar_follower_tweets_ids_df(tweets_data_df)
        tweets_A_df = tweets_proc_df[tweets_proc_df['author_id'].isin([user_id])]
        tweets_B_df = tweets_proc_df[tweets_proc_df['id'].isin(enc_ids)]
        vectors = TfidfVectorizer(min_df=1, stop_words="english", dtype=np.float32)
        cos_df = pd.DataFrame()

        if compare_by_tweet:
            cos_df = compute_similarity_vectorized(
                A=tweets_A_df,
                B=tweets_B_df,
                user_id=user_id,
                resp_ids=resp_ids,
                rows_in_slice=100, 
                vectors=vectors
            )
        else:
            cos_df = (
                tweets_B_df
                .groupby('author_id')
                .apply(
                    compute_similarity,
                    tweets_A_df['text_clean'].values,
                    tweets_A_df['id'],
                    resp_ids,
                    vectors,
                    compare_by_tweet
                )
            )  


            yield (cos_df.reset_index(drop=True), user_id)
    



def compute_similarity(group, doc_A, i1, i2, vectors):
    corpus = np.concatenate((doc_A, [' '.join(group["text_clean"].values)]))
    tfidf = vectors.fit_transform([' '.join(group["text_clean"].values)])
    tfidf = vectors.transform(corpus)
    cos_sim = cosine_similarity(tfidf)[:-1, -1].ravel()
    data = {
        'id_A': i1,
        'author_B_id': group.name,
        'cosine_similarity': cos_sim,
        'resp_cnt': sum(group['id'].isin([i2]))
    }
    return pd.DataFrame(data)


def compute_similarity_vectorized(A, B, user_id, resp_ids, rows_in_slice, vectors):
    A_doc = ' '.join(A['text_clean'].values)
    tfidf = vectors.fit_transform([A_doc])
    chunk_size = B.shape[0] 
    num_chunks = int(chunk_size / rows_in_slice)
    B_chunks = np.array_split(B, num_chunks)
    cos_df = pd.DataFrame()
    for chunk in B_chunks:
        corpus = np.concatenate((chunk['text_clean'].values, [A_doc]))
        tfidf = vectors.transform(corpus)
        cos_sim = cosine_similarity(tfidf)[:-1, -1].ravel()
        data = {
            'id_B': chunk['id'].values,
            'author_A_id': user_id,
            'cosine_similarity': cos_sim,
            'responded': chunk['id'].isin(resp_ids)
        }
        tmp_df = pd.DataFrame(data)
        cos_df = pd.concat([cos_df, tmp_df])

    return cos_df
    