import ast
import os
import library.const as con
import emoji
import pandas as pd
import re
import numpy as np
import xml.sax.saxutils as saxutils
import unicodedata

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
    s[['text_fixed_length','text_clean']] = tweet_length(s)

    return s


def handle_tweets_entities(s: pd.Series) -> pd.Series:
    link_exist , image_exist, hashtag_exist, quoted_exist = False, False, False, False
    link_cnt , image_cnt, hashtag_cnt = 0, 0, 0
    link_start, image_start, hashtag_start = [], pd.NA, []
    link_end, image_end, hashtag_end = [], pd.NA, []
    deleted_origin, quoted_start, quoted_end = False, pd.NA, pd.NA
    tmp_s = s
    val = 0
    x, y = [0,0],[0,0]
    if tmp_s['entities'] != 'None':
        entities_ast = ast.literal_eval(s['entities'])

        if 'urls' in entities_ast:
            for url in entities_ast['urls']:
                x[val], y[val] = url['start'], url['end']
                if con.TWITTER_URL in url['expanded_url']:
                    if 'photo' in url['expanded_url']:
                        if image_cnt == 0: 
                            image_exist = True
                            image_start, image_end = x[val], y[val]
                        image_cnt += 1
                    else:
                        quoted_exist = True
                        quoted_start, quoted_end = x[val], y[val]
                else:
                    if 't.co' in url['url'] and 't.co' in url['expanded_url']:
                        quoted_exist = True
                        deleted_origin = True
                        quoted_start, quoted_end = x[val], y[val]
                    else:
                        if link_cnt == 0: link_exist = True
                        link_start.append(x[val])
                        link_end.append(y[val])
                        link_cnt += 1
                val = 1 - val
            
        if 'hashtags' in entities_ast:
            for hashtag in entities_ast['hashtags']:
                x[val], y[val] = hashtag['start'], hashtag['end']
                if hashtag_cnt == 0: hashtag_exist = True
                hashtag_start.append(x[val])
                hashtag_end.append(y[val])
                hashtag_cnt += 1
                
    
    tmp_s[[
            'link_exist','link_count','link_start','link_end',
            'image_exist','image_count','image_start','image_end',
            'hashtag_exist','hashtag_count','hashtag_start','hashtag_end',
            'quoted_exist', 'quoted_start', 'quoted_end', 'deleted_origin'
        ]] = (
            link_exist, link_cnt, map_l(link_start), map_l(link_end),
            image_exist, image_cnt, image_start, image_end,
            hashtag_exist, hashtag_cnt, map_l(hashtag_start), map_l(hashtag_end),
            quoted_exist, quoted_start, quoted_end, deleted_origin
        )
    
    return tmp_s


def process_tweet_text_df(tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(columns=con.TWEET_TEXT_LIST)

    df[['id','author_id','type','text_raw','entities']] = tweets_df[['id','author_id','type','text','entities']].copy()
    df['text_raw_length'] = df['text_raw'].str.len()
    df = df.apply(restructure_tweet_text, axis=1)

    return df


def tweet_length(s: pd.Series) -> tuple[int, str]:
    length = 0
    txt = s['text_raw']

    # handle quoted link
    if s['quoted_exist']:
        txt = remove_substring(txt, s['quoted_start'], s['quoted_end'])
    
    # handle images link:
    if s['image_exist']:
        txt = remove_substring(txt, s['image_start'], s['image_end'])

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
    short_emoji = re.findall(r'©|®',txt)
    mod_cnt = count_emoji_modifier(txt)
    txt, cnt, cnt_u = clean_emoji(txt)
    length += (((cnt - mod_cnt) * 2) - len(short_emoji))

    # handle HTML entites in txt
    txt = saxutils.unescape(txt)

    # count rest of the text
    length += len(txt)

    return length, txt



def remove_substring(txt, idx_start, idx_stop):
    idx_s = idx_start - 1 if idx_start != 0 else 0
    text = txt[0: idx_s:] + txt[idx_stop + 1::]

    return text


def map_l(v):
    x = ','.join(map(str, v))

    return x


def checkEmojiType(strEmo):
    try:
        is_modifier = unicodedata.name(strEmo).startswith("EMOJI MODIFIER")
        if is_modifier: return None
        else: return strEmo
    except:
        return strEmo


def count_emoji_modifier(s):
    emoji_str = map_l(emoji.distinct_emoji_list(s))
    return len(list(c for c in emoji_str if c in emoji.EMOJI_DATA and checkEmojiType(c) is None))