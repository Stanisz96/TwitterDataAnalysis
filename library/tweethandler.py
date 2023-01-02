import os
import library.const as con

def create_emoji_dict(emoji_path: str) -> str:
    emoji_filenames = [
        f for f in os.listdir(emoji_path) 
        if os.path.isfile(os.path.join(emoji_path, f))
    ]

    print(emoji_filenames[0:20])

    return "test"