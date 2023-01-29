import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def tweets_len_factor(tweets_len_factor_df: pd.DataFrame):

    for i, group in tweets_len_factor_df.groupby('id'):
        plt.scatter(group['tweet_length'], group['resp_prob'], s=8, label=i, c='b')

    plt.xlabel('tweet length')
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
        loglog: bool=False,
        linear_reg: bool=False
    ):

    plt.scatter(data1[x1], data1[y1], s=10, label=label1)
    if data2 is not None:
        plt.scatter(data2[x2], data2[y2], s=10, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if loglog:
        plt.xscale('log')
        plt.yscale('log')

    if linear_reg:
        slope1, intercept1 = np.polyfit(data1[x1], data1[y1], 1)
        print(f'Slope for {label1}: {slope1}')
        reg_x1 = data1[x1]
        reg_y1 = slope1 * reg_x1 + intercept1
        plt.plot(reg_x1, reg_y1, color='red', label=f'Linear regression for {label1}')

        if x2 is not None:
            slope2, intercept2 = np.polyfit(data2[x2], data2[y2], 1)
            print(f'Slope for {label2}: {slope2}')
            reg_x2 = data2[x2]
            reg_y2 = slope2 * reg_x2 + intercept2
            plt.plot(reg_x2, reg_y2, color='green', label=f'Linear regression for {label1}')

    plt.legend()
    plt.title(title)
    plt.grid(True, linestyle='--')
    plt.show()

