from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .vocab import EOS_ID, PAD_ID, TigerVocab

MAX_HISTORY = 20


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
def load_interactions(path):
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=["user_id", "item_id", "timestamp"])
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        rows = []
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append((record["reviewerID"], record["asin"], record["unixReviewTime"]))
        df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def k_core(df, k):
    if k <= 1:
        return df
    while True:
        users = df.groupby("user_id").size()
        items = df.groupby("item_id").size()
        keep = df["user_id"].map(users).ge(k) & df["item_id"].map(items).ge(k)
        if keep.all():
            return df
        df = df[keep]


def build_sequences(df):
    df = df.reset_index(drop=True)
    order = np.argsort(df["timestamp"].to_numpy(), kind="stable")
    df = df.iloc[order]
    return {user: group.tolist() for user, group in df.groupby("user_id", sort=False)["item_id"]}


def load_project_split(processed_dir):
    processed = Path(processed_dir)
    cols = ["user_idx", "item_idx", "item_id", "timestamp"]
    frames = {s: pd.read_parquet(processed / f"{s}.parquet", columns=cols)
              for s in ("train", "validation", "test")}
    mapping = pd.concat([f[["item_idx", "item_id"]] for f in frames.values()]).drop_duplicates()
    mapping = mapping.sort_values("item_idx")
    if mapping.item_idx.duplicated().any() or mapping.item_id.duplicated().any():
        raise ValueError("item_idx and item_id are not one-to-one across the splits")
    if not np.array_equal(mapping.item_idx.to_numpy(), np.arange(len(mapping))):
        raise ValueError("item_idx is not contiguous from 0")
    train = frames["train"].sort_values(["user_idx", "timestamp"], kind="stable")
    sequences = {u: g.tolist() for u, g in train.groupby("user_idx", sort=False)["item_id"]}
    val = dict(zip(frames["validation"].user_idx, frames["validation"].item_id))
    test = dict(zip(frames["test"].user_idx, frames["test"].item_id))
    if not (set(sequences) == set(val) == set(test)):
        raise ValueError("users differ across the three split files")
    for user, seq in sequences.items():
        seq.append(val[user])
        seq.append(test[user])
    return sequences, mapping.item_id.astype(str).tolist()


def train_item_degree(sequences, index):
    degree = np.zeros(index.num_items, dtype=np.int64)
    for items in sequences.values():
        for item in items[:-2]:
            degree[index.id_to_idx[item]] += 1
    return degree


# --------------------------------------------------------------------------- #
# Semantic IDs
# --------------------------------------------------------------------------- #
def load_semantic_id_table(path):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    for prefix in ("c", "semantic_id_"):
        cols = sorted(
            (c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()),
            key=lambda c: int(c[len(prefix):]),
        )
        if cols:
            break
    else:
        raise ValueError("no code columns (c0.. or semantic_id_0..) found in the table")
    return df["item_id"].astype(str).tolist(), df[cols].to_numpy(dtype=np.int64)


def load_codes_npy(path, item_ids):
    codes = np.load(path)
    if codes.ndim != 2 or len(codes) != len(item_ids):
        raise ValueError(f"{path}: expected {len(item_ids)} rows, got shape {codes.shape}")
    return list(item_ids), codes.astype(np.int64)


def add_collision_codes(codes, collision_size=256):
    codes = np.asarray(codes, dtype=np.int64)
    _, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    seen = {}
    extra = np.zeros(len(codes), dtype=np.int64)
    for row, group in enumerate(inverse):
        extra[row] = seen.get(group, 0)
        seen[group] = extra[row] + 1
    if extra.max(initial=0) >= collision_size:
        raise ValueError(f"collision group larger than {collision_size}; enlarge the last level")
    return np.concatenate([codes, extra[:, None]], axis=1)


@dataclass
class ItemIndex:

    item_ids: list
    codes: np.ndarray  # int64 [N, num_levels], 0-based per-level codes
    vocab: TigerVocab

    def __post_init__(self):
        self.id_to_idx = {item: i for i, item in enumerate(self.item_ids)}
        if len(self.id_to_idx) != len(self.item_ids):
            raise ValueError("duplicate item_id in the Semantic-ID table")
        if len({tuple(row) for row in self.codes.tolist()}) != len(self.codes):
            raise ValueError("Semantic IDs are not unique; apply add_collision_codes first")
        offsets = np.array([self.vocab.level_offset(l) for l in range(self.vocab.num_levels)])
        tokens = self.codes + offsets
        # extra last row = padding item (all PAD tokens)
        pad_row = np.full((1, tokens.shape[1]), PAD_ID, dtype=np.int64)
        self.item_tokens = np.concatenate([tokens, pad_row], axis=0)

    @property
    def num_items(self):
        return len(self.item_ids)

    @property
    def pad_index(self):
        return self.num_items


def make_item_index(item_ids, codes, vocab=None, collision_size=256):
    vocab = vocab or TigerVocab()
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape[1] == vocab.num_levels - 1:
        codes = add_collision_codes(codes, collision_size)
    if codes.shape[1] != vocab.num_levels:
        raise ValueError(f"table has {codes.shape[1]} codes/item; vocab expects {vocab.num_levels}")
    for level, size in enumerate(vocab.level_sizes):
        if codes[:, level].min() < 0 or codes[:, level].max() >= size:
            raise ValueError(f"level {level} codes outside [0, {size})")
    return ItemIndex(list(item_ids), codes, vocab)


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #
@dataclass
class Examples:

    history: np.ndarray  # int64 [n, MAX_HISTORY] item indices
    target: np.ndarray  # int64 [n] item index
    user_token: np.ndarray  # int64 [n]
    user_ids: list  # raw ids, for reporting

    def __len__(self):
        return len(self.target)


def build_examples(sequences, index, vocab, split, max_history=MAX_HISTORY, min_len=5):
    hist, targ, utok, uids = [], [], [], []
    pad = index.pad_index
    for user, items in sequences.items():
        if len(items) < max(min_len, 3):
            continue
        idx = [index.id_to_idx[i] for i in items]  # KeyError => item missing a Semantic ID
        if split == "train":
            usable = idx[:-2]
            pairs = [(max(0, t - max_history), t) for t in range(1, len(usable))]
            seq = usable
        elif split == "val":
            seq, pairs = idx[:-1], [(max(0, len(idx) - 2 - max_history), len(idx) - 2)]
        elif split == "test":
            seq, pairs = idx, [(max(0, len(idx) - 1 - max_history), len(idx) - 1)]
        else:
            raise ValueError(split)
        token = index.vocab.user_token(user)
        for start, t in pairs:
            window = seq[start:t]
            hist.append(window + [pad] * (max_history - len(window)))
            targ.append(seq[t])
            utok.append(token)
            uids.append(user)
    return Examples(
        history=np.asarray(hist, dtype=np.int64).reshape(-1, max_history),
        target=np.asarray(targ, dtype=np.int64),
        user_token=np.asarray(utok, dtype=np.int64),
        user_ids=uids,
    )


def collate(examples, rows, index, torch_module, device, use_user_token=True):
    torch = torch_module
    cache = index.__dict__.setdefault("_device_tokens", {})
    if str(device) not in cache:
        cache[str(device)] = torch.as_tensor(index.item_tokens, device=device)
    item_tokens = cache[str(device)]
    hist = torch.as_tensor(examples.history[rows], device=device)
    target = torch.as_tensor(examples.target[rows], device=device)
    flat = item_tokens[hist].reshape(hist.size(0), -1)  # [B, 20 * levels]
    if use_user_token:
        user = torch.as_tensor(examples.user_token[rows], device=device).unsqueeze(1)
        flat = torch.cat([user, flat], dim=1)
    attention_mask = (flat != PAD_ID).long()
    labels = torch.cat(
        [item_tokens[target], torch.full((target.size(0), 1), EOS_ID, device=device)], dim=1
    )
    return flat, attention_mask, labels, target, hist
