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



def draw_average_tweets_len_factor(
        average_by_user_tweets_len_factor_df: pd.DataFrame,
        average_by_all_tweets_len_factor_df: pd.DataFrame
    ):

    user = average_by_user_tweets_len_factor_df
    all = average_by_all_tweets_len_factor_df

    plt.scatter(user['tweet_length'], user['resp_prob'], s=10, label='Average by user')
    plt.scatter(all['tweet_length'], all['resp_prob'], s=10, label='Average by all')
    plt.xlabel('tweet length')
    plt.ylabel('average response probability')
    plt.legend()
    plt.title('Average by user/all Response Probability vs Text Length')
    plt.grid(True, linestyle='--')
    plt.show()