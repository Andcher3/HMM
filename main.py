import os
os.environ["OMP_NUM_THREADS"] = "4"

from dataset import load_figure_dataset
from Model import model_lst
import numpy as np


def train(observations, labels, lens):
    HMM_models = []
    # print(model_lst, observations, labels, lens)
    for (model, observation, label, len) in zip(model_lst, observations, labels, lens):
        model.fit(observation, len)
        print(f"model {label} train finished.")
        HMM_models.append(model)
    return HMM_models


def predict(models, sample):
    scores = [model.score(sample) for model in models]
    return np.argmax(scores), scores


if __name__ == '__main__':
    (observations, kmeans_data, labels, lens), (test_data, test_kmeans, test_labels, test_lens) = (
        load_figure_dataset('data_figure', n_clusters=8))
    # print(observations)
    # print(labels)
    models = train(observations, labels, lens)
    err = 0
    for i, (samples, lens) in enumerate(zip(test_data, test_lens)):
        start = 0
        for sample_len in lens:
            pred, scores = predict(models, samples[start:start + sample_len])
            start += sample_len
            if pred != i:
                err += 1
    err /= np.sum([len(samples) for samples in test_data])
    print(err)
