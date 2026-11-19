# ================================================================
#  CONTRASTIVE PRE-TRAINING (GRACE-style)
#  Pre-trains the type projections + Metapath 1 GAT of the
#  MetapathGAT on graph structure alone, before any interaction
#  labels are introduced.
#
#  Memory note: the NT-Xent loss compares every node with every
#  other node ([2N, 2N] matrix). With all nodes of the full dataset
#  that is tens of GB, so each epoch uses a random sample of
#  PRETRAIN_SAMPLE_NODES nodes. Over 100 epochs every node is
#  still seen many times.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from config import HIDDEN_DIM, PRETRAIN_EPOCHS, PRETRAIN_SAMPLE_NODES


def augment(x, edge_index, drop_prob=0.2):
    """Create one augmented view: drop edges + add feature noise."""
    ei_aug, _ = dropout_edge(edge_index, p=drop_prob, training=True)
    x_aug     = x + torch.randn_like(x) * 0.01
    return x_aug, ei_aug


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent contrastive loss."""
    N  = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z  = torch.cat([z1, z2], dim=0)

    sim  = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)


def pretrain(model, hetero_data, edge_index, epochs=PRETRAIN_EPOCHS):
    """
    Run contrastive pre-training directly on the MetapathGAT's
    projection layers and Metapath 1 GAT (so no weight copying /
    shape mismatch afterwards).

    model       : MetapathGAT
    hetero_data : graph with node features
    edge_index  : flat direct lncRNA–miRNA edges (TRAINING edges only)
    """
    device    = edge_index.device
    projector = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, 64)
    ).to(device)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()), lr=0.001
    )

    print("── Contrastive Pre-training ──────────────────────")
    model.train()
    for epoch in range(epochs):
        optim.zero_grad()
        x = model.project_all_nodes(hetero_data)
        x1, ei1 = augment(x, edge_index)
        x2, ei2 = augment(x, edge_index)
        h1 = model.run_gat(x1, ei1, model.gat_direct_1, model.gat_direct_2)
        h2 = model.run_gat(x2, ei2, model.gat_direct_1, model.gat_direct_2)

        n   = x.shape[0]
        idx = torch.randperm(n, device=device)[:min(PRETRAIN_SAMPLE_NODES, n)]
        loss = nt_xent_loss(projector(h1[idx]), projector(h2[idx]))
        loss.backward()
        optim.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}  Contrastive Loss: {loss.item():.4f}")

    print("── Pre-training complete ─────────────────────────\n")
    return model
