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

def count_emoji(text: str) -> int:
    cnt = emoji.emoji_count(text)

    return cnt


def process_tweet_text_df(tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=con.TWEET_TEXT_LIST)





