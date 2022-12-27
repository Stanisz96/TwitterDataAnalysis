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

    # Load every individual data, count tweets text length
    # and save data to tweets_text_len_count
    if step_number == 3:

        tweets_df_gen = fo.load_by_one_all_individual()
        tweets_text_len_count_df = proc.tweets_text_len(tweets_df_gen)
        fo.save_data(
            f'{con.PROC_PATH}/tweets_text_len_count',
            tweets_text_len_count_df,
            True
        )



if __name__=='__main__':
    main(3)