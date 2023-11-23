from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans

from fllib.utils import torch_utils
from .aggregators import Mean, Median


class Signguard(object):
    r"""A robust aggregator from paper `Xu et al.

    SignGuard: Byzantine-robust Federated
    Learning through Collaborative Malicious Gradient
    Filtering <https://arxiv.org/abs/2109.05872>`_.
    """

    def __init__(self, agg="mean", max_tau=1e5, linkage="average") -> None:
        super(Signguard, self).__init__()

        assert linkage in ["average", "single"]
        self.tau = max_tau
        self.linkage = linkage
        self.l2norm_his = []
        if agg == "mean":
            self.agg = Mean()
        elif agg == "median":
            self.agg = Median()
        else:
            raise NotImplementedError(f"{agg} is not supported yet.")

    def __call__(self, inputs: List[torch.Tensor]):
        updates = torch.stack(inputs, dim=0)
        num = len(updates)
        l2norms = [torch.norm(update).item() for update in updates]
        M = np.median(l2norms)

        for idx in range(num):
            if l2norms[idx] > M:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], M)
                l2norms[idx] = torch.norm(updates[idx]).item()
        L = 0.1
        R = 3.0
        S1_idxs = []
        for idx, (l2norm, update) in enumerate(zip(l2norms, updates)):
            if l2norm >= L * M and l2norm <= R * M:
                S1_idxs.append(idx)

        features = []
        num_para = len(updates[0])
        for update in updates:
            feature0 = (update > 0).sum().item() / num_para
            feature1 = (update < 0).sum().item() / num_para
            feature2 = (update == 0).sum().item() / num_para

            features.append([feature0, feature1, feature2])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

        flag = 1 if np.sum(kmeans.labels_) > num // 2 else 0
        S2_idxs = list(
            [idx for idx, label in enumerate(kmeans.labels_) if label == flag]
        )

        inter = list(set(S1_idxs) & set(S2_idxs))
        benign_updates = []
        for idx in inter:
            if l2norms[idx] > M:
                updates[idx] = torch_utils.clip_tensor_norm_(updates[idx], M)
            benign_updates.append(updates[idx])

        values = self.agg(benign_updates)
        return values
