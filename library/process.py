from typing import Generator
import pandas as pd
import collections as col
import library.restructure as res
import library.const as con
import matplotlib.pyplot as plt

def create_aggregated_df(tweets_df_gen: Generator[pd.DataFrame, None, None], column_name: str) -> pd.DataFrame:
    aggregated_df = pd.DataFrame()

    for tweets_df in tweets_df_gen:
        tmp_df = (
            tweets_df[f'{column_name}']
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
