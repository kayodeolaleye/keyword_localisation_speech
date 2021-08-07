import os
import sys

BASE_PATH = "Interspeech"
sys.path.append(BASE_PATH)
import config


def load(path, parser):
    with open(path, "r") as f:
        return list(map(parser, f.readlines()))


def parse_token(line):
    key, *words = line.strip().split()
    text = " ".join(words)
    img, i = key.split("#")
    key1 = img.split(".")[0] + "_" + str(i)
    return key1, text


def wav_path_to_key(wav_path):
    _, filename = os.path.split(wav_path)
    key, _ = os.path.splitext(filename)
    return key


def wav_to_img_path(path):
    key = wav_path_to_key(path)
    key_img = get_key_img(key)
    return os.path.join(
        config.BASE_DIR,
        "flickr8k-images",
        "Flicker8k_Dataset",
        key_img + ".jpg",
    )


def get_key_img(key):
    *parts, _ = key.split("_")
    return "_".join(parts)
