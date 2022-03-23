import pdb

from train_emb import (
    Flickr8kDataset,
)


def read(path, parse=lambda row: row.strip()):
    with open(path, "r") as f:
        return list(map(parse, f.readlines()))


keywords = read("data/flickr8k/vocab-1000.txt")
keywords_seen = read("data/flickr8k/vocab-67-seen.txt")

keywords = list(set(keywords) - set(keywords_seen))

samples = Flickr8kDataset.load_samples("test")
transcripts = Flickr8kDataset.load_transcripts()

counts = {word: 0 for word in keywords}

for sample in samples:
    for word in transcripts[sample.value].lower().split():
        if word in keywords:
            counts[word] += 1

counts = sorted(counts.items(), reverse=True, key=lambda kv: kv[1])

with open("data/flickr8k/vocab-933-by-freq-flickr8-test.txt", "w") as f:
    for k, v in counts:
        f.write("{} {}\n".format(k, v))
