import library.fileoperations as fo
import library.process as proc
import library.const as con
import library.restructure as res
import library.draw as draw
import library.tweethandler as th
import matplotlib.pyplot as plt
import pandas as pd
import emoji 
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def main(step_number: int):
    # Testing
    if step_number == 0:

        print("Welcome to Twitter data analysis project!")


    # Restructure data to individual form and save them to feather format.
    # Where individual refers to data for one following user and 
    # all data of users that this user is following.
    if step_number == 1:

        fo.save_all_tweets_individuals()


    # Restructure data containing users informations and
    # save them to feather format.
    if step_number == 2:

        users_data_df = res.get_users_dataframe()
        
        fo.save_data(
            f'{con.DATA_PATH}/users',
            users_data_df,
            True
        )


    # Load individual data, perform restructuring tweets data
    # and save new dataFromat containing attributes related to tweets text
    if step_number == 3:
        tweets_df_gen = fo.load_by_one_all_individual(con.DATA_PATH)
        for df in tweets_df_gen:
            tmp_df = th.process_tweet_text_df(df)
            id = df['author_id'].iat[0]
            fo.save_data(
                f'{con.PROC_PATH}/tweets/{str(id)}',
                tmp_df,
                True
            )


    # Load individual data, perform aggregating data, save data
    # and draw histogram
    if step_number == 4:
        tweets_df_gen = fo.load_by_one_all_individual(con.PROC_PATH)
        aggregated_df = proc.create_aggregated_df(tweets_df_gen, ['text_raw_length', 'text_fixed_length'])
        aggregated_df.plot(y=['text_raw_length', 'text_fixed_length'], x='len_values')
        plt.show()


    # First factor -> tweet length
    if step_number == 5:
        # Load data
        tweets_proc_df_gen = fo.load_by_one_all_individual(con.PROC_PATH)
        tweets_data_df_gen = fo.load_by_one_all_individual(con.DATA_PATH)

        # Process data
        tweets_len_factor_df = proc.tweets_len_factor(tweets_proc_df_gen, tweets_data_df_gen)
        avg_by_user_tweets_len_factor_df = proc.average_by_user_tweets_len_factor(tweets_len_factor_df)
        avg_by_all_tweets_len_factor_df = proc.average_by_all_tweets_len_factor(tweets_len_factor_df)

        # Save data
        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/avg_by_user_tweets_len_factor',
                avg_by_user_tweets_len_factor_df,
                True
        )
        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/avg_by_all_tweets_len_factor',
                avg_by_all_tweets_len_factor_df,
                True
        )


    # Calculate correlations for tweet lengt factor
    if step_number == 6:
        avg_by_all_tweets_len_factor_df = fo.load_data(
            f'{con.PROC_PATH}/factor_tweet_length/avg_by_all_tweets_len_factor'
        )
        avg_by_user_tweets_len_factor_df = fo.load_data(
            f'{con.PROC_PATH}/factor_tweet_length/avg_by_user_tweets_len_factor'
        )
        avg_by_all_tweets_len_factor_df = avg_by_all_tweets_len_factor_df.query("resp_prob != 1")
        avg_by_user_tweets_len_factor_df = avg_by_user_tweets_len_factor_df.query("resp_prob != 1")
        avg_all_140_df = avg_by_all_tweets_len_factor_df.query("tweet_length < 140")
        avg_all_280_df = avg_by_all_tweets_len_factor_df.query("tweet_length >= 140")
        avg_user_140_df = avg_by_user_tweets_len_factor_df.query("tweet_length < 140")
        avg_user_280_df = avg_by_user_tweets_len_factor_df.query("tweet_length >= 140")

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_by_all_tweets_len_factor',
                avg_by_all_tweets_len_factor_df.corr().reset_index(), True
        )

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_by_user_tweets_len_factor',
                avg_by_user_tweets_len_factor_df.corr().reset_index(), True
        )

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_all_140',
                avg_all_140_df.corr().reset_index(), True
        )

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_all_280',
                avg_all_280_df.corr().reset_index(), True
        )

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_user_140',
                avg_user_140_df.corr().reset_index(), True
        )

        fo.save_data(
                f'{con.PROC_PATH}/factor_tweet_length/correlation/avg_user_280_df',
                avg_user_280_df.corr().reset_index(), True
        )


    # Second factor -> cosine similarity using TF-IDF
    # Compare similarity of data on user A to users B
    if step_number == 7:
        # Load data
        tweets_proc_df_gen = fo.load_by_one_all_individual(con.PROC_PATH, data_type='en')
        global_df_gen = fo.load_by_one_all_individual(con.PROC_PATH, data_type='en')
        tweets_data_df_gen = fo.load_by_one_all_individual(con.DATA_PATH, return_id = True, data_type='en')

        # Process data  
        cos_sim_df = proc.cosine_similarity_factor(tweets_proc_df_gen, tweets_data_df_gen, global_df_gen, 'prod', 0)

        # Draw data
        draw.scatter_results(
            data1=cos_sim_df,
            label1='Cosine similarity',
            title='Cosine similarity user A vs user B',
            x1='cosine_similarity',
            y1='resp_prob',
            xlabel='cosine similarity',
            ylabel='response probability'
        )

        # Save data
        fo.save_data(
            f'{con.PROC_PATH}/cos_similarity/by_users/data_filtered_en',
            cos_sim_df.reset_index(drop=True),
            True
        )

    if step_number == 8:
        df = fo.load_data(
            f'{con.PROC_PATH}/cos_similarity/by_users/data_filtered_en'
        )          
        print(df.info())
        draw.scatter_results(
            data1=df,
            label1='Cosine similarity',
            title='Cosine similarity user A vs user B',
            x1='cosine_similarity',
            y1='resp_prob',
            xlabel='cosine similarity',
            ylabel='response probability',
            loglog=True
        )

    # Filter data and save only english tweets
    if step_number == 9:
        tweets_proc_df_gen = fo.load_by_one_all_individual(con.PROC_PATH)
        tweets_data_df_gen = fo.load_by_one_all_individual(con.DATA_PATH)

        df_gen = res.get_english_data_gen(tweets_proc_df_gen, tweets_data_df_gen)
        for proc_df, data_df in df_gen:
            id_data = data_df['author_id'].iat[0]
            id_proc = data_df['author_id'].iat[0]
            fo.save_data(
                f'{con.PROC_PATH}/tweets/en/{str(id_data)}',
                proc_df.reset_index(drop=True),
                True
            )
            fo.save_data(
                f'{con.DATA_PATH}/tweets/en/{str(id_proc)}',
                data_df.reset_index(drop=True),
                True
            )


    # Initialize final data to store information regarding factors
    if step_number == 10:
        tweets_data_df_gen = fo.load_by_one_all_individual(con.DATA_PATH, data_type='en', return_id=True)
        final_gen = res.create_final_data_gen(tweets_data_df_gen, return_id = True)

        for final_df, id in final_gen:
            fo.save_data(
                f'{con.PROC_PATH}/final/{str(id)}',
                final_df,
                True
            )

    # Test
    if step_number == 11:
        tweets_final_df_gen = fo.load_by_one_all_individual(con.PROC_PATH, data_type='final')
        cnt = 0
        for tweets_final_df in tweets_final_df_gen:
            tmp_cnt = tweets_final_df['created_at_A'].count()
            if tmp_cnt > 0:
                cnt += tmp_cnt
        print(f'All resonses cnt: {cnt}')


if __name__=='__main__':
    main(11)


