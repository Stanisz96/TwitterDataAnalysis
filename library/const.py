import pathlib as pl
import numpy as np

test = True

if test:
    RAW_DATA_PATH = pl.Path('C:/Magisterka/dataV2')
    RAW_USERS_PATH = pl.Path(RAW_DATA_PATH,"users")
    DATA_PATH = pl.Path('./test/data')
    PROC_PATH = pl.Path('./test/data_processed')
    IMAGES_PATH = pl.Path('./test/images')
else:
    RAW_DATA_PATH = pl.Path('C:/Magisterka/dataV2')
    RAW_USERS_PATH = pl.Path(RAW_DATA_PATH,"users")
    DATA_PATH = pl.Path('./data')
    PROC_PATH = pl.Path('./data_processed')
    IMAGES_PATH = pl.Path('./images')

RAW_DATA_PATH_TEST = pl.Path('C:/Magisterka/dataTest')
RAW_USERS_PATH_TEST = pl.Path(RAW_DATA_PATH_TEST,"users")
DATA_PATH_TEST = pl.Path('./test/data')
PROC_PATH_TEST = pl.Path('./test/data_processed')
IMAGES_PATH_TEST = pl.Path('./test/images')
TWITTER_URL = 'https://twitter.com'

TWEET_TYPE_NAMES = [
    'quoted',
    'replied_to',
    'retweeted',
    'tweeted'
    ]
TWEET_COLUMN_NAMES = [
    'author_id',
    'id',
    'text',
    'created_at',
    'lang',
    'source',
    'type',
    'ref_id',
    'retweet_count',
    'reply_count',
    'like_count',
    'quote_count',
    'downloaded_date_time',
    'conversation_id',
    'entities'
    ]
TWEET_TYPES_LIST = {
    'author_id': np.uint64,
    'id': np.uint64,
    'text': str,
    'created_at': str,
    'lang': str,
    'source': str,
    'type': str,
    'ref_id': np.uint64,
    'retweet_count': np.uint32,
    'reply_count': np.uint32,
    'like_count': np.uint32,
    'quote_count': np.uint32,
    'downloaded_date_time': str,
    'conversation_id': np.uint64,
    'entities': str
    }
USER_COLUMN_NAMES = [
    'type',
    'id',
    'name',
    'username',
    'created_at',
    'description',
    'followers_count',
    'tweet_count',
    'verified',
    'protected'
    ]
USER_TYPES_LIST = {
    'type': str,
    'id': np.uint64,
    'name': str,
    'username': str,
    'created_at': str,
    'description': str,
    'followers_count': np.uint32,
    'tweet_count': np.uint32,
    'verified': bool,
    'protected': bool
    }
COUNT_USERS_RESP_COLUMN_NAMES = [
    "user_id",
    "encountered_count",
    "stranger_resp_count",
    "familiar_resp_count",
    "quoted_count",
    "replied_to_count",
    "retweeted_count",
    "tweeted_count",
    "familiar_resp_prob",
    "quoted_prob",
    "replied_to_prob",
    "retweeted_prob",
    "tweeted_prob"
    ]
COUNT_USERS_RESP_TYPE_LIST = {
    'user_id': np.uint64,
    'encountered_count': np.uint32,
    'stranger_resp_count': np.uint32, 
    'familiar_resp_count': np.uint32,
    'quoted_count': np.uint32,
    'replied_to_count': np.uint32,
    'retweeted_count': np.uint32,
    'tweeted_count': np.uint32,
    'familiar_resp_prob': np.float32,
    'quoted_prob': np.float32,
    'replied_to_prob': np.float32,
    'retweeted_prob': np.float32,
    'tweeted_prob': np.float32
    }
TWEET_TEXT_LIST = {
    'id': np.uint64,
    'author_id': np.uint64,
    'type': str,
    'text_raw': str,
    'text_no_emoji': str,
    'text_clean': str,
    'text_raw_length': np.uint16,
    'text_fixed_length': np.uint16,
    'emoji_count': np.uint32,
    'unique_emoji_count': np.uint16,
    'link_exist': bool,
    'link_count': np.uint16,
    'link_start': str,
    'link_end': str,
    'image_exist': bool,
    'image_count': np.uint16,
    'image_start': np.uint16,
    'image_end': np.uint16,
    'hashtag_exist': bool,
    'hashtag_count': np.uint16,
    'hashtag_start': str,
    'hashtag_end': str,
    'quoted_exist': bool,
    'quoted_start': np.uint16,
    'quoted_end': np.uint16,
    'entities': str
}

