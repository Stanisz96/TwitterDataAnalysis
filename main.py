import library.fileoperations as fo
import library.process as proc
import library.const as con
import library.restructure as res

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


    # Load every individual data, count tweets text length,
    # clean tweets text and save to data_processed
    if step_number == 3:

        tweets_df_gen = fo.load_by_one_all_individual(con.DATA_PATH)
        fo.save_all_tweets_individuals_cleaned(tweets_df_gen)


    # Load all individuals, count occurance of tweets length
    # and save data to feather format
    if step_number == 4:
        tweets_df_gen = fo.load_by_one_all_individual(con.PROC_PATH)
        tweets_length_count_df = proc.tweets_text_len(tweets_df_gen)
        fo.save_data(
            f'{con.PROC_PATH}/all_tweets_text_len_count',
            tweets_length_count_df,
            True
        )


if __name__=='__main__':
    # main(4)
    print('rrr')