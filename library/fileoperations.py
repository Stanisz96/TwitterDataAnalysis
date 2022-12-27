import os
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


def save_data_frame_to_feather(path: str, data_frame: pd.DataFrame):
    '''Save data frame to feather format, using given path.'''

    create_folders_for_path(path, True)
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


def save_users_data():
    '''
    Save DataFrame objects, containing users data from userData.json files.
    Saved data are in feather format.
    '''

    path = f'{con.DATA_PATH}/users'
    users_data_df = res.get_users_dataframe()
    save_data_frame_to_feather(path, users_data_df)

