import os
import library.const as con
import emoji
import pandas as pd


def create_emoji_dict(emoji_path: str) -> str:
    emoji_filenames = [
        f for f in os.listdir(emoji_path) 
        if os.path.isfile(os.path.join(emoji_path, f))
    ]

    print(emoji_filenames[0:20])

    return "test"


def clean_emoji(text: str) -> str:
    clean_text = emoji.replace_emoji(text, '')

    return clean_text

def count_emoji(text: str, unique: bool = False) -> int:
    cnt = emoji.emoji_count(text, unique)

    return cnt

def tweet_length(text: str) -> int:
    tweet_len = len(text)
    return tweet_len

def handle_clean_text(s: pd.Series) -> pd.DataFrame:
    s['text_clean'] = clean_emoji(s['text_raw'])
    s['text_clean_length'] = len(s['text_clean'])
    s['emoji_count'] = count_emoji(s['text_raw'])
    s['unique_emoji_count'] = count_emoji(s['text_raw'], True)
    s['text_fixed_length'] = tweet_length(s['text_raw'])

    return s

def process_tweet_text_df(tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=con.TWEET_TEXT_LIST)

    df[['id','text_raw']] = tweets_df[['id','text']].copy()
    df['text_raw_length'] = df['text_raw'].str.len()
    df = df.apply(handle_clean_text, axis=1)

    print(df)




