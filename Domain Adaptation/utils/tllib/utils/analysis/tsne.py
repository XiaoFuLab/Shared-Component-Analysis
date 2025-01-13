#credits https://github.com/thuml/Transfer-Learning-Library
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b', labels= None):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    plt.close('all')
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    if labels is not None:
        # visualize using matplotlib
        view1 = X_tsne[: X_tsne.shape[0] // 2, ]
        view2 = X_tsne[X_tsne.shape[0] // 2:, ]
        if isinstance(labels, list):
            labels_1 = labels[0]
            labels_2 = labels[1]
        else:
            labels_1 = labels
            labels_2 = labels
            
        colors = cm.rainbow(np.linspace(0, 1, len(labels_1.unique())))
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.scatter(view1[:, 0], view1[:, 1], c=colors[labels_1], marker='v', s=20)
        plt.scatter(view2[:, 0], view2[:, 1], c=colors[labels_2], marker='o', s=20)
        # plt.xticks([])
        # plt.yticks([])
        if filename is None:
            return plt.gcf()
        else:
            plt.savefig(filename)
            plt.close()
    else:
        domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

        # visualize using matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
        # plt.xticks([])
        # plt.yticks([])
        if filename is None:
            return plt.gcf()
        else:
            plt.savefig(filename)
            plt.close()
