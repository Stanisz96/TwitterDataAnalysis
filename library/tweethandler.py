import ast
import os
import library.const as con
import emoji
import pandas as pd
import re
import json


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


def restructure_tweet_text(s: pd.Series) -> pd.DataFrame:
    s[['text_no_emoji', 'emoji_count', 'unique_emoji_count']] = clean_emoji(s['text_raw'])
    s[['link_exist','link_count','link_start','link_end',
       'image_exist','image_count','image_start','image_end',
       'hashtag_exist','hashtag_count','hashtag_start','hashtag_end'
    ]] = handle_tweets_entities(s)
    s[['text_fixed_length','text_clean']] = tweet_length(s['text_raw'], s['image_exist'])

    return s


def handle_tweets_entities(s: pd.Series) -> pd.Series:
    link_exist , image_exist, hashtag_exist = False, False, False
    link_cnt , image_cnt, hashtag_cnt = 0, 0, 0
    link_start, image_start, hashtag_start = pd.NA, pd.NA, pd.NA
    link_end, image_end, hashtag_end = pd.NA, pd.NA, pd.NA
    tmp_s = s

    if tmp_s['entities'] != 'None':
        entities_ast = ast.literal_eval(s['entities'])
        if 'urls' in entities_ast:
            for url in entities_ast['urls']:
                x, y = str(url['start']), str(url['end'])
                if con.TWITTER_URL in url['expanded_url']:
                    if image_cnt == 0: 
                        image_exist = True
                        image_start, image_end = x, y
                    else:
                        image_start += f',{x}'
                        image_end += f',{y}'
                    image_cnt += 1
                else:
                    if link_cnt == 0: 
                        link_exist = True
                        link_start, link_end = x, y
                    else:
                        link_start += f',{x}'
                        link_end += f',{y}'
                    link_cnt += 1
            
        if 'hashtags' in entities_ast:
            for hashtag in entities_ast['hashtags']:
                x, y = str(hashtag['start']), str(hashtag['end'])
                if hashtag_cnt == 0: 
                    hashtag_exist = True
                    hashtag_start, hashtag_end = x, y
                else:
                    hashtag_start += f',{x}'
                    hashtag_end += f',{y}'
                hashtag_cnt += 1
                
    
    tmp_s[[
            'link_exist','link_count','link_start','link_end',
            'image_exist','image_count','image_start','image_end',
            'hashtag_exist','hashtag_count','hashtag_start','hashtag_end'
        ]] = (
            link_exist, link_cnt, link_start, link_end,
            image_exist, image_cnt, image_start, image_end,
            hashtag_exist, hashtag_cnt, hashtag_start, hashtag_end
        )
    
    return tmp_s


def process_tweet_text_df(tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=con.TWEET_TEXT_LIST)

    df[['id', 'author_id','text_raw','entities']] = tweets_df[['id','author_id','text','entities']].copy()
    df['text_raw_length'] = df['text_raw'].str.len()
    df = df.apply(restructure_tweet_text, axis=1)

    return df


def tweet_length(text: str, image_exist: bool = False) -> tuple[int, str]:
    length = 0
    txt = text

    # handle links
    if image_exist:
        txt, links_cnt = re.subn(r'https://.{15}','', text)
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