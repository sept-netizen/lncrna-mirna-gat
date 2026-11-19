# ================================================================
#  TRAINING — fine-tunes MetapathGAT on labeled interaction data
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from config import (FINETUNE_EPOCHS, LEARNING_RATE, NEG_RATIO,
                    MAX_POS_PER_EPOCH)


def pair_codes(pairs, n_mir):
    """Encode (lnc, mir) pairs as single integers lnc * n_mir + mir."""
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return arr[:, 0] * n_mir + arr[:, 1]


def sample_negatives(pos_codes, n_lnc, n_mir, n):
    """
    Sample n random lncRNA–miRNA pairs that are NOT known interactions.
    pos_codes : sorted numpy array from pair_codes() of all positives
    Vectorised with numpy so it stays fast with millions of pairs.
    """
    out, have = [], 0
    while have < n:
        l = np.random.randint(0, n_lnc, size=2 * n)
        m = np.random.randint(0, n_mir, size=2 * n)
        ok = ~np.isin(l * n_mir + m, pos_codes)
        out.append(np.stack([l[ok], m[ok]], axis=1))
        have += ok.sum()
    return np.concatenate(out)[:n]


def train(model, hetero_data, direct_edges, metapath_edges,
          train_pos, pos_codes, n_lnc, n_mir):
    """
    Fine-tune the MetapathGAT on lncRNA–miRNA interaction labels.
    direct_edges must contain ONLY training interactions, otherwise
    the model can "see" the test answers through the graph.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn   = nn.BCEWithLogitsLoss()
    train_pos = np.asarray(train_pos, dtype=np.int64)

    print("── Fine-tuning ───────────────────────────────────")
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Encode all nodes
        embeddings = model.encode(hetero_data, direct_edges, metapath_edges)

        # Fresh random subset of positives (if there are very many)
        if len(train_pos) > MAX_POS_PER_EPOCH:
            pick = np.random.choice(len(train_pos), MAX_POS_PER_EPOCH,
                                    replace=False)
            pos = train_pos[pick]
        else:
            pos = train_pos
        neg = sample_negatives(pos_codes, n_lnc, n_mir, len(pos) * NEG_RATIO)

        all_pairs = torch.from_numpy(np.concatenate([pos, neg]))
        labels    = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])

        scores = model.predict(embeddings, all_pairs[:, 0], all_pairs[:, 1], n_lnc)
        loss   = loss_fn(scores, labels.to(scores.device))
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}  Loss: {loss.item():.4f}")

    print("── Fine-tuning complete ──────────────────────────\n")
    return model
