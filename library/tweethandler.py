import os
import library.const as con
import emoji
import pandas as pd
import re

def create_emoji_dict(emoji_path: str) -> str:
    emoji_filenames = [
        f for f in os.listdir(emoji_path) 
        if os.path.isfile(os.path.join(emoji_path, f))
    ]

    print(emoji_filenames[0:20])

    return "test"


def clean_emoji(text: str) -> tuple[str, int, int]:
    clean_text = emoji.replace_emoji(text, '')
    cnt = emoji.emoji_count(text, False)
    cnt_u = emoji.emoji_count(text, True)

    return clean_text, cnt, cnt_u


def handle_clean_text(s: pd.Series) -> pd.DataFrame:
    s[['text_no_emoji', 'emoji_count', 'unique_emoji_count']] = clean_emoji(s['text_raw'])
    s['text_clean_length'] = len(s['text_clean'])
    s[['text_fixed_length','text_clean']] = tweet_length(s['text_raw'])

    return s

def process_tweet_text_df(tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=con.TWEET_TEXT_LIST)

    df[['id','text_raw']] = tweets_df[['id','text']].copy()
    df['text_raw_length'] = df['text_raw'].str.len()
    df = df.apply(handle_clean_text, axis=1)

    print(df)




def tweet_length(s: pd.Series) -> tuple[int, str]:
    length = 0

    # handle links
    txt, links_cnt = re.subn(r'https://.{15}','', s['text'])
    length += (links_cnt * 23)

    # handle retweet
    tmp_len = len(txt)
    txt = re.sub(r'^RT @.*?: ', '', txt)
    length += (tmp_len - len(txt))

    # handle mentions
    mentions_cnt = txt.count('@')
    for i in range(mentions_cnt):
        found = re.search(r'^@.*?\s', txt)
        mention_idx = txt.find('@')

        if found and mention_idx != -1:
            txt = txt[mention_idx:]\
                        [txt.find(' ')+1:]
        else:
            break

    # handle emoji
    txt, cnt, cnt_u = clean_emoji(txt)
    length += (cnt * 2)

    # count rest of the text
    length += len(txt)

    return length, txt