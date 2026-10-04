"""Next-basket datasets, following van Maasakkers, Fok & Donkers (2023), github.com/luukvanmaasakkers/nextbasketpredictionGRU.

Each dataset becomes a list of users, each a chronological list of baskets (arrays of item indices).
The last basket of every user is held out; users are split 50/50 into validation and test. All
earlier baskets of every user are available for training.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class BasketData:
    name: str
    n_items: int
    baskets: list
    gaps: list
    valid_users: np.ndarray
    test_users: np.ndarray
    item_names: dict = field(default_factory=dict)

    def history(self, u):
        return self.baskets[u][:-1]

    def target(self, u):
        return self.baskets[u][-1]

    def history_items(self, u):
        return np.unique(np.concatenate(self.history(u)))


def _split_users(n_users, seed):
    perm = np.random.default_rng(seed).permutation(n_users)
    half = n_users // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def _to_baskets(df, user_col, basket_col, item_col, time_cols, gap_col=None):
    df = df.sort_values([user_col] + time_cols + [basket_col], kind="stable")
    items = df[item_col].to_numpy()
    basket_change = np.flatnonzero(np.diff(df[basket_col].to_numpy())) + 1
    b_start = np.concatenate([[0], basket_change])
    b_end = np.concatenate([basket_change, [len(df)]])
    b_user = df[user_col].to_numpy()[b_start]
    b_gap = df[gap_col].to_numpy(dtype=np.float32)[b_start] if gap_col else np.zeros(len(b_start), np.float32)
    user_change = np.flatnonzero(np.diff(b_user)) + 1
    baskets_flat = [np.unique(items[s:e]) for s, e in zip(b_start, b_end)]
    bounds = np.concatenate([[0], user_change, [len(b_start)]])
    baskets = [baskets_flat[s:e] for s, e in zip(bounds[:-1], bounds[1:])]
    gaps = [b_gap[s:e] for s, e in zip(bounds[:-1], bounds[1:])]
    return baskets, gaps


def cluster_rare_products(purchases, products, threshold=500):
    """Python port of cluster_products_instacart.R: within each aisle, rare products (fewer than
    `threshold` purchases) are greedily merged with the most similar rare cluster (cosine similarity
    of user purchase vectors) until every cluster reaches the threshold or the aisle runs out of
    rare products. Frequent products keep their own index. Returns product_id -> cluster id."""
    counts = purchases.groupby("product_id").size()
    users = purchases["user_id"].astype("category")
    prods = purchases["product_id"].astype("category")
    m = sp.csr_matrix((np.ones(len(purchases), dtype=np.float32), (prods.cat.codes, users.cat.codes)))
    row_of = {p: i for i, p in enumerate(prods.cat.categories)}
    aisle = products.set_index("product_id")["aisle_id"]
    mapping, next_id = {}, 0
    for p in counts.index[counts >= threshold]:
        mapping[p] = next_id
        next_id += 1
    rare = counts[counts < threshold]
    for _, members in rare.groupby(aisle.reindex(rare.index).to_numpy()):
        ids = list(members.index)
        clusters = [[p] for p in ids]
        sizes = members.to_numpy().astype(np.int64)
        vecs = sp.vstack([m[row_of[p]] for p in ids]).tocsr()
        dots = (vecs @ vecs.T).toarray().astype(np.float64)
        alive = np.ones(len(ids), dtype=bool)
        while alive.sum() > 1:
            small = np.where(alive & (sizes < threshold))[0]
            if len(small) == 0:
                break
            a = small[np.argmin(sizes[small])]
            diag = np.diag(dots)
            sim = dots[a] / np.sqrt(diag[a] * diag + 1e-12)
            sim[~alive] = -np.inf
            sim[a] = -np.inf
            b = int(np.argmax(sim))
            clusters[b].extend(clusters[a])
            sizes[b] += sizes[a]
            alive[a] = False
            dots[b] += dots[a]
            dots[:, b] += dots[:, a]
        for i in np.where(alive)[0]:
            for p in clusters[i]:
                mapping[p] = next_id
            next_id += 1
    return pd.Series(mapping, name="cluster")


def load_instacart(raw_dir, cache_dir, user_frac=1.0, cluster_threshold=500, seed=123):
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache = cache_dir / f"instacart_u{user_frac}_t{cluster_threshold}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    orders = pd.read_csv(raw_dir / "orders.csv")
    orders = orders[orders["eval_set"] != "test"]
    op = pd.concat([pd.read_csv(raw_dir / "order_products__prior.csv", usecols=["order_id", "product_id"]),
                    pd.read_csv(raw_dir / "order_products__train.csv", usecols=["order_id", "product_id"])])
    df = op.merge(orders[["order_id", "user_id", "order_number", "days_since_prior_order"]], on="order_id")
    products = pd.read_csv(raw_dir / "products.csv")
    history_only = df[df["order_number"] < df.groupby("user_id")["order_number"].transform("max")]
    clusters = cluster_rare_products(history_only, products, cluster_threshold)
    df = df[df["product_id"].isin(clusters.index)]
    df["item"] = clusters.reindex(df["product_id"]).to_numpy()
    if user_frac < 1.0:
        users = np.sort(df["user_id"].unique())
        keep = np.random.default_rng(seed).choice(users, int(len(users) * user_frac), replace=False)
        df = df[df["user_id"].isin(keep)]
    df = df[df.groupby("user_id")["order_id"].transform("nunique") >= 3]
    df["days_since_prior_order"] = df["days_since_prior_order"].fillna(0.0)
    baskets, gaps = _to_baskets(df, "user_id", "order_id", "item", ["order_number"], "days_since_prior_order")
    valid, test = _split_users(len(baskets), seed)
    names = products.set_index("product_id")["product_name"]
    item_names = {}
    for p, c in clusters.items():
        item_names.setdefault(int(c), names.get(p, str(p)))
    data = BasketData("instacart", int(clusters.max()) + 1, baskets, gaps, valid, test, item_names)
    pd.to_pickle(data, cache)
    return data


def load_dunnhumby(raw_dir, cache_dir, min_item_count=50, min_baskets=3, max_baskets=100, max_items=None, seed=123):
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache = cache_dir / f"dunnhumby_c{min_item_count}_b{max_baskets}_i{max_items}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_csv(raw_dir / "transaction_data.csv", usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "TRANS_TIME"])
    counts = df.groupby("PRODUCT_ID").size()
    keep = counts[counts >= min_item_count].sort_values(ascending=False)
    if max_items is not None:
        keep = keep.iloc[:max_items]
    df = df[df["PRODUCT_ID"].isin(keep.index)]
    df = df[df.groupby("household_key")["BASKET_ID"].transform("nunique") >= min_baskets]
    order = df.drop_duplicates("BASKET_ID").sort_values(["household_key", "DAY", "TRANS_TIME", "BASKET_ID"])
    order["rank_from_end"] = order.groupby("household_key").cumcount(ascending=False)
    recent = order.loc[order["rank_from_end"] < max_baskets, "BASKET_ID"]
    df = df[df["BASKET_ID"].isin(recent)]
    item_index = {p: i for i, p in enumerate(np.sort(df["PRODUCT_ID"].unique()))}
    df["item"] = df["PRODUCT_ID"].map(item_index)
    order = order[order["BASKET_ID"].isin(recent)].copy()
    order["gap"] = order.groupby("household_key")["DAY"].diff().fillna(0.0).clip(upper=50)
    df = df.merge(order[["BASKET_ID", "gap"]], on="BASKET_ID")
    baskets, gaps = _to_baskets(df, "household_key", "BASKET_ID", "item", ["DAY", "TRANS_TIME"], "gap")
    valid, test = _split_users(len(baskets), seed)
    data = BasketData("dunnhumby", len(item_index), baskets, gaps, valid, test)
    pd.to_pickle(data, cache)
    return data


def basket_stats(data):
    n_b = np.array([len(b) for b in data.baskets])
    sizes = np.concatenate([[len(x) for x in b] for b in data.baskets])
    rep = []
    for u in range(len(data.baskets)):
        t = data.target(u)
        rep.append(np.isin(t, data.history_items(u)).mean())
    return {
        "Users": len(data.baskets),
        "Items": data.n_items,
        "Baskets": int(n_b.sum()),
        "Mean baskets per user": round(float(n_b.mean()), 2),
        "Mean basket size": round(float(sizes.mean()), 2),
        "Repeat ratio (target)": round(float(np.mean(rep)), 3),
        "Valid users": len(data.valid_users),
        "Test users": len(data.test_users),
    }
