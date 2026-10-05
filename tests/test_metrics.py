import math
from functools import partial

import numpy as np
import pytest
import torch

from kanrec.metrics import basket_user_metrics, session_metrics_from_ranks, summarize, target_ranks


close = partial(math.isclose, rel_tol=1e-6)


def test_target_ranks_are_pessimistic_on_ties():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.9], [0.3, 0.2, 0.1, 0.0]])
    ranks = target_ranks(scores, torch.tensor([1, 0]))
    assert ranks.tolist() == [2, 1]


def test_session_metrics_hand_computed():
    m = session_metrics_from_ranks([1, 3, 15, 50], cutoffs=(10, 20))
    assert math.isclose(m["Recall@10"], 2 / 4)
    assert math.isclose(m["Recall@20"], 3 / 4)
    assert math.isclose(m["MRR@20"], (1 + 1 / 3 + 1 / 15) / 4)
    assert math.isclose(m["NDCG@10"], (1 + 1 / math.log2(4)) / 4)


def test_basket_metrics_hand_computed():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    truth = [np.array([1, 4])]
    history = [np.array([1, 2])]
    m = summarize(basket_user_metrics(scores, truth, history, cutoffs=(2, 5)))
    assert close(m["Recall@2"], 0.5)
    assert close(m["Recall@5"], 1.0)
    assert m["PHR@2"] == 1.0
    assert close(m["NDCG@2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert close(m["NDCG@5"], (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3)))
    assert close(m["P@B"], 0.5)
    assert close(m["P@B/2"], 0.0)
    assert close(m["R@2B"], 0.5)
    assert close(m["AvgRank"], (2 + 5) / 2)
    assert close(m["RepRecall@2"], 1.0)
    assert close(m["ExplRecall@2"], 0.0)
    assert close(m["ExplRecall@5"], 1.0)


def test_vectorised_basket_metrics_match_loop():
    from kanrec.metrics import basket_user_metrics_loop
    rng = np.random.default_rng(0)
    U, N = 40, 60
    scores = torch.tensor(rng.integers(0, 8, (U, N)).astype(np.float32))
    truth = [rng.choice(N, rng.integers(1, 12), replace=False) for _ in range(U)]
    hist = [rng.choice(N, rng.integers(1, 20), replace=False) for _ in range(U)]
    a = basket_user_metrics_loop(scores, truth, hist)
    b = basket_user_metrics(scores, truth, hist)
    assert set(a) == set(b)
    for k in a:
        assert np.allclose(a[k], b[k], equal_nan=True), k


def test_sparse_bce_matches_dense():
    import torch.nn.functional as F
    from kanrec.basket import _multi_hot, _target_index, sparse_bce
    rng = np.random.default_rng(1)
    steps = [[rng.choice(30, rng.integers(1, 6), replace=False) for _ in range(rng.integers(1, 5))] for _ in range(4)]
    T = max(len(s) for s in steps)
    logits = torch.randn(4, T, 30)
    y = _multi_hot(steps, 30, torch.device("cpu"))
    dense = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean(-1)
    rows, cols = _target_index(steps, T, torch.device("cpu"))
    assert torch.allclose(sparse_bce(logits, rows, cols), dense, atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple GPU")
def test_vectorised_basket_metrics_on_mps():
    rng = np.random.default_rng(2)
    scores = torch.tensor(rng.random((10, 30), dtype=np.float32))
    truth = [rng.choice(30, 3, replace=False) for _ in range(10)]
    hist = [rng.choice(30, 5, replace=False) for _ in range(10)]
    cpu = basket_user_metrics(scores, truth, hist)
    mps = basket_user_metrics(scores.to("mps"), truth, hist)
    for k in cpu:
        assert np.allclose(cpu[k], mps[k], equal_nan=True), k
