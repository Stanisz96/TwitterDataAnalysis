import os
from typing import Generator
import pandas as pd
import library.process as proc
import library.const as con
import library.restructure as res

def create_folders_for_path(path: str, with_file: bool = False):
    '''
    Check if path exists. If not, then create necessary folders.

    * with_file : boolean
        define if given path contains file name
    '''

    path_folder = path
    if with_file:
        path_folder = path[:path.rfind('/')]

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)


def save_data(path: str, data_frame: pd.DataFrame, with_file: bool):
    '''Save data frame to feather format, using given path.'''

    create_folders_for_path(path, with_file)
    data_frame.to_feather(path)


def save_all_tweets_individuals():
    '''
    Save DataFrame objects, containing tweets data for every individual.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    Saved data are in feather format and filename is user following id.
    '''

    users_ids_df : pd.DataFrame = res.get_users_ids()
    users_following_ids_df  = users_ids_df[users_ids_df['type'] == 'A']
    folder_path = f'{con.DATA_PATH}/tweets'
    create_folders_for_path(folder_path)

    for index, row in users_following_ids_df.iterrows():
        path = f'{folder_path}/{row["id"]}'
        if os.path.exists(path):
            users_following_ids_df = users_following_ids_df.drop(index=[index])
    counter = 0

    for tweets_df in res.gen_tweets_dataframes(users_following_ids_df):
        if tweets_df.empty:
            print(f'UserId: {users_following_ids_df.iloc[counter]}')
        else:
            id = tweets_df.iloc[0]['author_id']
            tweets_df.to_feather(f'{folder_path}/{id}')

        counter += 1


def save_all_tweets_individuals_cleaned(tweets_individual_gen: Generator[list, None, None]):
    '''
    Same as save_all_tweets_individuals(), but with cleaned text and text_length col.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    Saved data are in feather format and filename is user following id.
    '''

    for tweets_individual in tweets_individual_gen:
        df = res.get_individual_clean_tweets_text_df(tweets_individual)
        id = df.iloc[0]['author_id']
        folder_path = f'{con.PROC_PATH}/tweets'
        create_folders_for_path(folder_path)
        df.to_feather(f'{folder_path}/{id}')
    

##########################################################################################

def load_by_one_all_individual(main_path: str):
    users_ids_df = res.get_users_ids()
    users_following_ids_df = users_ids_df[users_ids_df['type'] == 'A']
    loaded = 0
    not_loaded = 0
    for index, row in users_following_ids_df.iterrows():
        id = row['id']
        path = f'{main_path}/tweets/{id}'
        if os.path.exists(path):
            loaded += 1
            yield pd.read_feather(f'{main_path}/tweets/{id}')
        else:
            not_loaded += 1
    
    print(f'Loaded users A: {loaded} / {loaded+not_loaded}')
    print(f'Not loaded users A: {not_loaded} / {loaded+not_loaded}')



def load_follower_users_data():
    '''
    Load feather format file and return DataFrame object containing follower users data.
    '''

    users_df: pd.DataFrame = pd.read_feather(f'{con.DATA_PATH}/users')
    follower_users_df = users_df.query("type == 'A'")

    return follower_users_df


def load_data(path: str):
    data_df = pd.read_feather(path)
    return data_df