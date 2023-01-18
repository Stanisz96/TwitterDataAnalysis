from typing import Generator
import pandas as pd
import collections as col
import library.restructure as res
import library.const as con
import library.fileoperations as fo
import matplotlib.pyplot as plt
import numpy as np

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




# def tweets_len_factor(
#     tweets_df_gen: Generator[pd.DataFrame, None, None]
#     ) -> pd.DataFrame:

