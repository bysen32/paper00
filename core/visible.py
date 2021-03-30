# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    outputs = np.load(f'tools/inter_data/outputs_1.npy', allow_pickle=True)
    data = outputs[0]
    data = np.stack(data)
    label = outputs[1]
    # label = np.array([i[0] for i in label])

    # outputs = np.load(f'tools/inter_data/outputs_1.npy', allow_pickle=True)
    # ood_outputs = np.load(f'tools/inter_data/outputs_ood_1.npy', allow_pickle=True)
    # data_1 = outputs[0]
    # data_1 = np.stack(data_1)
    # label_1 = outputs[1]
    # data_2 = ood_outputs[0]
    # data_2 = np.stack(data_2)
    # label_2 = ood_outputs[1]

    # data = np.concatenate((data_1, data_2), axis=0)
    # label = np.concatenate((np.ones_like(label_1), np.zeros_like(label_2)))
    return data, label


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    # plt.show(fig)
    plt.show()


if __name__ == '__main__':
    main()