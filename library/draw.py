import matplotlib.pyplot as plt
import pandas as pd

def draw_tweets_len_factor(tweets_len_factor_df: pd.DataFrame):

    for i, group in tweets_len_factor_df.groupby('id'):
        plt.scatter(group['tweet_length'], group['resp_prob'], s=8, label=i, c='b')

    plt.xlabel('tweet dlength')
    plt.ylabel('response probability')
    plt.title('Response Probability vs Text Length')
    plt.grid(True, linestyle='--')
    plt.show()



def scatter_results(
        data1: pd.DataFrame,
        x1: str,
        y1: str,
        label1: str,
        xlabel: str,
        ylabel: str,
        title: str,
        data2: pd.DataFrame=None,
        x2: str=None,
        y2: str=None,
        label2: str=None,
    ):

    plt.scatter(data1[f'{x1}'], data1[f'{y1}'], s=10, label=label1)
    if data2 is not None:
        plt.scatter(data2[f'{x2}'], data2[f'{y2}'], s=10, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--')
    plt.show()