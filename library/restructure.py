import ast
import pathlib as pl
from typing import Generator
import library.const as con
import numpy as np
import pandas as pd
import os
import json
import re


def get_users_ids() -> pd.DataFrame:
    '''
    Get all users ids and type from userList file.
    Return DataFrame object with two columns: type, id.
    '''
    
    lines = []

    with open(pl.Path(con.RAW_DATA_PATH,"usersList.dat")) as file:
        for line in file:
            line = line.strip()
            lines.append([line[0],line[1:]])

    users_ids_array = np.array(lines)
    users_ids_df = pd.DataFrame(users_ids_array, columns = ['type','id'])\
                    .reset_index()

    return users_ids_df


def gen_tweets_dataframes(users_following_ids_df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
    '''
    Generate DataFrame objects for every individual.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    '''
    
    for index, row in users_following_ids_df.iterrows(): 
        yield pd.DataFrame(list(gen_tweets_array(row['id'])),
                           columns=con.TWEET_COLUMN_NAMES)\
                                .astype(con.TWEET_TYPES_LIST)\
                                .reset_index()


def gen_tweets_array(user_following_id: np.uint64) -> Generator[list, None, None]:
    '''
    Generate array of tweets data converted from json file for individual.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    '''

    users_ids_array = get_all_ids_for_individual(user_following_id)

    for users_following_id in users_ids_array:
        tweets_path = pl.Path(con.RAW_USERS_PATH,str(users_following_id),"tweets")

        for idx, tweet_type in enumerate(con.TWEET_TYPE_NAMES):
            tweets_type_path = pl.Path(tweets_path,tweet_type)

            for json_file in os.listdir(tweets_type_path): 
                json_path = os.path.join(tweets_type_path, json_file)
                if os.path.isfile(json_path):
                    json_temp = json.load(open(json_path, encoding="utf8"))
                    tweet_fix_type, ref_id = fix_tweet_type(
                        json_temp['Entities'],
                        json_temp['Referenced_tweets'][0]['Type'],
                        json_temp['Referenced_tweets'][0]['Id']
                    )

                    yield [json_temp['Author_id'],json_temp['Id'],
                           json_temp['Text'],json_temp['Created_at'],
                           json_temp['Lang'],json_temp['Source'],
                           tweet_fix_type, ref_id,
                           json_temp['Public_metrics']['Retweet_count'],
                           json_temp['Public_metrics']['Reply_count'],
                           json_temp['Public_metrics']['Like_count'],
                           json_temp['Public_metrics']['Quote_count'],
                           json_temp['DownloadedDateTime'],
                           json_temp['Conversation_id'],
                           json_temp['Entities']]


def get_all_ids_for_individual(user_following_id: np.uint64) -> list:
    '''
    Get all ids for individual user given in fun parm.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    '''

    ids_array = [user_following_id]
    ids_array.extend(get_following_ids_of_an_user(user_following_id))

    return ids_array


def get_following_ids_of_an_user(user_following_id: np.uint64) -> list:
    '''
    Get all ids of users that are followed by user given in fun parm.
    '''

    json_path = os.path.join(
        con.RAW_USERS_PATH,
        user_following_id,
        "metaData.json"
        )
    json_file = json.load(open(json_path, encoding="utf8"))
    following_ids_of_an_user = list(json_file['Following'])

    return following_ids_of_an_user


def get_users_dataframe() -> pd.DataFrame:
    '''
    Get users DataFrame object.
    Function return DataFrame containing all users data from userData.json files.
    '''
    
    users_dataframe = pd.DataFrame(list(gen_users_data_array(get_users_ids())),
                           columns=con.USER_COLUMN_NAMES)\
                                .astype(con.USER_TYPES_LIST)\
                                .reset_index()

    return users_dataframe


def gen_users_data_array(users_ids_df: pd.DataFrame):
    '''
    Generate list containing all users data.
    '''

    for index, row in users_ids_df.iterrows():
        json_path = pl.Path(con.RAW_USERS_PATH,row['id'],"userData.json")
        json_temp = json.load(open(json_path, encoding="utf-8"))
        if row['type'] == 'A':
            json_temp = json_temp['data']
        
            yield [row['type'], json_temp['id'], json_temp['name'], json_temp['username'],
                json_temp['created_at'], json_temp['description'],
                json_temp['public_metrics']['followers_count'],
                json_temp['public_metrics']['tweet_count'],
                json_temp['verified'],json_temp['protected']]
        
        if row['type'] == 'B':
            yield [row['type'], json_temp['Id'], json_temp['Name'], json_temp['Username'],
                json_temp['Created_at'], json_temp['Description'],
                json_temp['Public_metrics']['Followers_count'],
                json_temp['Public_metrics']['Tweet_count'],
                json_temp['Verified'],json_temp['Protected']]



def get_individual_clean_tweets_text_df(tweets_individual: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean tweets text in individual data and add length column.
    Where individual refers to data for one following user and 
    all data of users that this user is following.
    '''   

    tweets_individual['text'] = tweets_individual.apply(
                                    lambda row: remove_front_mentions(row['text']),
                                    axis=1)

    tweets_individual['text_length'] = tweets_individual['text'].str.len()
    tweets_individual = tweets_individual.astype({'text_length': np.uint32})

    return tweets_individual


def remove_front_mentions(text: str):
    count = text.count('@')
    for i in range(count):
        found = re.search(r'^@.*?\s', text)
        mention_index = text.find('@')

        if found and mention_index != -1:
            text = text[mention_index:]\
                        [text.find(' ')+1:]
        else:
            break

    return text


def fix_tweet_type(entities: json, tweet_type: str, ref_id: np.uint64) -> tuple[str, np.int64]:
    if entities is None:
        return tweet_type, ref_id
    if 'urls' in entities:
        for url in entities['urls']:
            if url['url'] == url['expanded_url']:
                return 'quoted', 0
    
    return tweet_type, ref_id



def familiar_follower_tweets_ids_df(
    tweets_df: pd.DataFrame,
    only_resp: bool=False
    ):
    df = tweets_df[['author_id','created_at','ref_id','id']].copy()
    df['created_at'] = pd.to_datetime(df['created_at'])
    user_A_id = df['author_id'].iloc[0]
    tweets_A_df = df.query('author_id == @user_A_id')
    familiar_A_resp_df = tweets_A_df.merge(
        df, 
        left_on='ref_id',
        right_on='id',
        suffixes=('_A', '_B'),
        )
    familiar_A_resp_df = familiar_A_resp_df.query("author_id_B != @user_A_id")

    if only_resp:
        return familiar_A_resp_df['id_B'].values
    else:
        first_A_time = tweets_A_df['created_at'].min()
        familiar_A_encountered_df = df.query(
            "created_at >= @first_A_time and author_id != @user_A_id"
        )
        return (
            familiar_A_resp_df['id_B'].values,
            familiar_A_encountered_df['id'].values
        )



def get_english_data_gen(
    tweets_proc_df_gen: Generator[pd.DataFrame, None, None],
    tweets_data_df_gen: Generator[pd.DataFrame, None, None]
    ) -> tuple[Generator[pd.DataFrame, None, None], Generator[pd.DataFrame, None, None]]:


    for proc_df, data_df in zip(tweets_proc_df_gen, tweets_data_df_gen):
        data_tmp = data_df[data_df['lang'].isin(['en'])]
        proc_tmp = proc_df[proc_df['id'].isin(data_tmp['id'])]

        yield (proc_tmp, data_tmp)


def create_final_data_gen(tweets_data_df_gen: Generator[pd.DataFrame, None, None], return_id: bool=False):

    for data_df, user_A_id in tweets_data_df_gen:
        author_A_id = np.uint64(user_A_id)
        tmp_A_df = data_df[data_df['author_id'].isin([author_A_id])]
        tmp_B_df = data_df[~data_df['author_id'].isin([author_A_id])]
        
        final_df = pd.merge(tmp_A_df, tmp_B_df, left_on='ref_id', right_on='id', how='right', suffixes=['_A','_B'])

        columns_to_keep = np.array(list(con.FINAL_DATA_LIST.keys()))
        fillna_dict = {'created_at_A': 'NaT', 'author_id_A': author_A_id, 'id_A': 0, 'source_A': 'None', 'type_A': 'None'}

        final_df = final_df[columns_to_keep]
        final_df.fillna(value=fillna_dict, inplace=True)
        final_df['created_at_A'] = pd.to_datetime(final_df['created_at_A'], errors='coerce')
        final_df = final_df.astype(con.FINAL_DATA_LIST)

        if return_id:
            yield final_df.reset_index(drop=True), author_A_id
        else:
            yield final_df.reset_index(drop=True)
        




def extend_final_with_tweet_length_gen(
    final_df_gen: Generator[pd.DataFrame, None, None],
    proc_df_gen: Generator[pd.DataFrame, None, None]
    ) -> tuple[pd.DataFrame, str]:

    for (final_df, author_id_A), proc_df in zip(final_df_gen, proc_df_gen):
        tweet_length = proc_df.loc[proc_df['id'].isin(final_df['id_B'].values), 'text_fixed_length']
        final_df['tweet_length_B'] = tweet_length.astype(np.uint16)

        yield final_df, author_id_A