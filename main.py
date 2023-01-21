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

        

if __name__=='__main__':
    main(6)
    
