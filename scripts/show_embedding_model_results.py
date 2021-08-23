import sys

import torch

import numpy as np
import streamlit as st

BASE_PATH = "Interspeech"
sys.path.append(BASE_PATH)
from train_emb import FeaturesImageCLIPLoader, Flickr8kDataset, get_key_img


class FeaturesAudioCLIPLoader:
    def __init__(self):
        path = "Interspeech/output/cnn-transformer-features-image-clip-flickr8k-test.npz"
        data = np.load(path)
        samples = Flickr8kDataset.load_samples("test")
        self.name_to_index = {n: i for i, n in enumerate(samples)}
        self.pred = data["pred"]

    def __call__(self, sample_name):
        index = self.name_to_index[sample_name]
        return self.pred[index]


def audio_to_image_retrieval(to_shuffle=True):
    transcripts = Flickr8kDataset.load_transcripts()
    samples = Flickr8kDataset.load_samples("test")
    samples_img = sorted(set([get_key_img(s) for s in samples]))

    if to_shuffle:
        import random
        random.shuffle(samples)

    audio_loader = FeaturesAudioCLIPLoader()
    image_loader = FeaturesImageCLIPLoader(BASE_PATH)

    audio_emb = torch.vstack([torch.tensor(audio_loader(s)) for s in samples])
    image_emb = torch.vstack([torch.tensor(image_loader(s + "_0")) for s in samples_img])
    
    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    sim = audio_emb @ image_emb.T
    sim = sim.numpy()

    st.set_page_config(layout="wide")
    st.title("retrieval · audio → image")
    st.markdown("""
        - for each audio utterance in the test set, retrieve the top five most similar images
        - use audio model trained in the embedding space with images represented by CLIP features
    """)

    for i, sample in enumerate(samples[:64]):
        top5 = sim[i].argsort()[::-1][:5]
        is_correct = any(get_key_img(sample) == samples_img[r] for r in top5)

        desc = [
            "text:     " + transcripts[sample],
            "in-top-5: " + str(is_correct),
        ]

        col, *_ = st.beta_columns([0.4, 0.6])
        col.markdown("## " + sample)
        col.image(Flickr8kDataset.get_image_path(sample))
        col.audio(Flickr8kDataset.get_audio_path(sample))
        col.code("\n".join(desc))

        st.markdown("top 5 images by audio query")
        cols = st.beta_columns(5)
        for r, col in zip(top5, cols):
            s = samples_img[r]
            is_correct = s == get_key_img(sample)
            t = "\n".join("\t" + transcripts[s + "_{}".format(i)] for i in range(5))
            desc = [
                "sample:     " + s,
                "score:      {:.3f}".format(sim[i, r]),
                "is-correct: " + str(is_correct),
                "transcripts:" + "\n" + t,
            ]
            col.image(Flickr8kDataset.get_image_path(s + "_0"))
            col.code("\n".join(desc))
        st.markdown("---")


def main():
    audio_to_image_retrieval()


if __name__ == "__main__":
    main()
