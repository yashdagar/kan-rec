import numpy as np
import pandas as pd
import pytest
import torch

from kanrec.session import GRU4RecModel, KANGRU4RecModel, evaluate, train_session_model


def test_plain_wrapper_matches_official_model():
    torch.manual_seed(0)
    ref = GRU4RecModel(50, [16], dropout_p_hidden=0.2)
    ours = KANGRU4RecModel(50, [16], dropout_p_hidden=0.2)
    ours.load_state_dict(ref.state_dict())
    X = torch.randint(0, 50, (4,))
    Y = torch.randint(0, 50, (9,))
    H1, H2 = [torch.randn(4, 16)], None
    H2 = [H1[0].clone()]
    assert torch.allclose(ref(X, H1, Y), ours(X, H2, Y))
    assert torch.allclose(H1[0], H2[0])


@pytest.mark.parametrize("kwargs", [
    dict(head_block="kan"), dict(head_block="mlp"), dict(input_block="kan"),
    dict(cells=["kangru"]), dict(cells=["kanrnn"]), dict(cells=["gru", "kanrnn"]),
])
def test_variants_forward_backward(kwargs):
    layers = [16, 16] if len(kwargs.get("cells", [])) == 2 else [16]
    m = KANGRU4RecModel(50, layers, **kwargs)
    H = [torch.zeros(4, h) for h in layers]
    out = m(torch.randint(0, 50, (4,)), H, torch.randint(0, 50, (12,)), training=True)
    assert out.shape == (4, 12)
    out.sum().backward()


def _toy_sessions(n_sessions, n_items, seed):
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    for s in range(n_sessions):
        start = rng.integers(0, n_items)
        for k in range(rng.integers(2, 6)):
            rows.append((s, (start + k) % n_items, t))
            t += 1.0
    return pd.DataFrame(rows, columns=["SessionId", "ItemId", "Time"])


def test_training_learns_deterministic_sequence():
    train = _toy_sessions(400, 30, 0)
    valid = _toy_sessions(50, 30, 1)
    params = dict(loss="cross-entropy", layers=[32], batch_size=16, learning_rate=0.1, n_sample=0,
                  dropout_p_hidden=0.0, constrained_embedding=True, sample_alpha=0.5, logq=0.0)
    model, imap, hist, layers = train_session_model(train, valid, params, torch.device("cpu"), n_epochs=3, log=lambda *_: None)
    assert hist[-1]["Recall@10"] > 0.9
    ranks = evaluate(model, valid, imap, layers, torch.device("cpu"))
    assert len(ranks) == int((valid.groupby("SessionId").size() - 1).sum())


def test_mlp_gru_cell_is_param_matched():
    from kanrec.kan import n_params
    from kanrec.session import KANGRUCell, MLPGRUCell
    kan, mlp = KANGRUCell(64, 64), MLPGRUCell(64, 64)
    assert abs(n_params(kan) - n_params(mlp)) / n_params(kan) < 0.02


def test_mlp_rnn_cell_is_param_matched():
    from kanrec.kan import n_params
    from kanrec.session import KANRNNCell, MLPRNNCell
    kan, mlp = KANRNNCell(64, 64), MLPRNNCell(64, 64)
    assert abs(n_params(kan) - n_params(mlp)) / n_params(kan) < 0.02


def test_device_iterator_matches_official():
    from kanrec.session import DeviceSessionIterator, SessionDataIterator, _reset_hidden
    data = _toy_sessions(120, 40, 3)
    outs = []
    for cls in (SessionDataIterator, DeviceSessionIterator):
        it = cls(data.copy(), 8, n_sample=0, device=torch.device("cpu"))
        H = [torch.zeros(8, 4)]
        hook = lambda n_valid, fm, vm: _reset_hidden(H, fm, vm, n_valid)
        outs.append([(a.tolist(), b.tolist()) for a, b in it(enable_neg_samples=False, reset_hook=hook)])
    assert outs[0] == outs[1] and len(outs[0]) > 30


def test_kannbr_batched_scores_match_per_user():
    from kanrec.data_basket import BasketData
    from kanrec.kannbr import KANNBR, NBRFeaturizer, kannbr_scores
    rng = np.random.default_rng(0)
    baskets = [[rng.choice(25, rng.integers(1, 5), replace=False) for _ in range(rng.integers(3, 8))] for _ in range(12)]
    data = BasketData("toy", 25, baskets, [np.zeros(len(b)) for b in baskets], np.arange(6), np.arange(6, 12))
    feat = NBRFeaturizer(data, n_popular=5)
    model = KANNBR(7, "kan").eval()
    out = kannbr_scores(model, feat, data, np.arange(12), torch.device("cpu"))
    for u in range(12):
        cand, X, _ = feat.user(data.history(u))
        with torch.no_grad():
            ref = model(torch.from_numpy(X))
        assert torch.allclose(out[u, torch.from_numpy(cand)], ref, atol=1e-5)
        other = np.setdiff1d(np.arange(25), cand)
        assert (out[u, torch.from_numpy(other)] == -1e9).all()
