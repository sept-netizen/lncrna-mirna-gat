# ================================================================
#  MODEL — MetapathGAT with type-aware node projections
#
#  Architecture:
#    1. TypeProjection: per-node-type linear → shared HIDDEN_DIM
#    2. Metapath 1 GAT: direct lncRNA–miRNA edges
#    3. Metapath 2 GAT: real lncRNA→Protein→miRNA metapath edges
#    4. Semantic attention: learned combination of both metapaths
#    5. MLP link prediction head
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import (
    RNA_FEATURE_DIM, GENERIC_FEATURE_DIM,
    HIDDEN_DIM, GAT_HEADS, DROPOUT, MLP_DROPOUT
)


class TypeProjection(nn.Module):
    """
    Addition 3: Node type-aware feature projection.

    Each node type has its own linear layer projecting its
    raw features into the shared HIDDEN_DIM embedding space.
    Followed by LayerNorm and ReLU for stable training.

    Why: lncRNA features (68-dim RNA descriptors) are not
    comparable to protein features (16-dim random init) without
    type-specific transformation first.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return F.relu(self.norm(self.proj(x)))


class MetapathGAT(nn.Module):
    """
    Metapath-Aware Heterogeneous Graph Attention Network.

    Metapath 1 — Direct edges:    lncRNA ↔ miRNA
    Metapath 2 — Protein-mediated: lncRNA → Protein → miRNA
                 (real metapath instances from lncrna_protein
                  and mirna_protein files, Addition 4)

    Each metapath runs through its own two-layer GAT.
    Semantic attention combines the two into a final embedding.
    MLP head predicts interaction probability for a pair.
    """

    def __init__(self, n_lnc, n_mir, n_pro, n_dis):
        super().__init__()

        # ── Type-specific projections (Addition 3) ───────────────
        # lncRNA and miRNA: project from 68-dim RNA features
        self.proj_lncrna  = TypeProjection(RNA_FEATURE_DIM, HIDDEN_DIM)
        self.proj_mirna   = TypeProjection(RNA_FEATURE_DIM, HIDDEN_DIM)
        # protein and disease: project from 16-dim generic features
        self.proj_protein = TypeProjection(GENERIC_FEATURE_DIM, HIDDEN_DIM)
        self.proj_disease = TypeProjection(GENERIC_FEATURE_DIM, HIDDEN_DIM)

        # ── Metapath 1: Direct lncRNA–miRNA GAT ──────────────────
        # Layer 1: GAT_HEADS heads of HIDDEN_DIM // GAT_HEADS dims,
        # concatenated back to HIDDEN_DIM. (8 full 128-dim heads would
        # store 1024 numbers per edge and run out of RAM on big graphs.)
        head_dim = HIDDEN_DIM // GAT_HEADS
        self.gat_direct_1 = GATConv(
            HIDDEN_DIM, head_dim,
            heads=GAT_HEADS, dropout=DROPOUT, concat=True
        )
        self.gat_direct_2 = GATConv(
            HIDDEN_DIM, HIDDEN_DIM,
            heads=1, dropout=DROPOUT, concat=False
        )

        # ── Metapath 2: Protein-mediated GAT (Addition 4) ────────
        self.gat_meta_1 = GATConv(
            HIDDEN_DIM, head_dim,
            heads=GAT_HEADS, dropout=DROPOUT, concat=True
        )
        self.gat_meta_2 = GATConv(
            HIDDEN_DIM, HIDDEN_DIM,
            heads=1, dropout=DROPOUT, concat=False
        )

        # ── Semantic attention ────────────────────────────────────
        # Learns which metapath matters more per node
        self.semantic_fc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.semantic_q  = nn.Parameter(torch.randn(HIDDEN_DIM))

        # ── MLP link prediction head ──────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(64, 1)
        )

        self.dropout_p = DROPOUT

    def project_all_nodes(self, hetero_data):
        """
        Project all node types into shared HIDDEN_DIM space.
        Concatenated in order: [lncRNA | miRNA | protein | disease]
        This flat matrix is what the GAT layers operate on.
        """
        h_lnc = self.proj_lncrna( hetero_data['lncrna' ].x)
        h_mir = self.proj_mirna(  hetero_data['mirna'  ].x)
        h_pro = self.proj_protein(hetero_data['protein'].x)
        h_dis = self.proj_disease(hetero_data['disease'].x)
        return torch.cat([h_lnc, h_mir, h_pro, h_dis], dim=0)

    def run_gat(self, x, edge_index, gat1, gat2):
        """
        Run one metapath's two-layer GAT on the flat node matrix.
        Returns updated node embeddings [N_total, HIDDEN_DIM].
        If edge_index is empty, returns x unchanged.
        """
        if edge_index.shape[1] == 0:
            # No edges for this metapath — return projected features
            return x

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.elu(gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = gat2(x, edge_index)
        return x

    def semantic_attention(self, z_list):
        """
        Combine metapath embeddings via learned semantic attention.

        For each metapath embedding Z_i:
          score_i = q · tanh(W · Z_i)   per node
          alpha_i = softmax(scores)

        Final = sum(alpha_i * Z_i)

        The model learns: for this node, which metapath gives
        more useful context for interaction prediction.
        """
        scores = []
        for z in z_list:
            s = torch.tanh(self.semantic_fc(z))    # [N, HIDDEN_DIM]
            scores.append((s * self.semantic_q).sum(-1))  # [N]

        # [N, num_metapaths]
        weights = torch.softmax(torch.stack(scores, dim=1), dim=1)

        out = torch.zeros_like(z_list[0])
        for i, z in enumerate(z_list):
            out = out + weights[:, i:i+1] * z
        return out

    def encode(self, hetero_data, direct_edge_index, metapath_edge_index):
        """
        Full encoding pass over the heterogeneous graph.

        1. Project all node types → shared space  [N_total, HIDDEN_DIM]
        2. Metapath 1 GAT (direct edges)          → Z1
        3. Metapath 2 GAT (protein-mediated)      → Z2
        4. Semantic attention                      → Z_final

        Returns Z_final [N_total, HIDDEN_DIM]
        """
        x  = self.project_all_nodes(hetero_data)
        z1 = self.run_gat(x, direct_edge_index,   self.gat_direct_1, self.gat_direct_2)
        z2 = self.run_gat(x, metapath_edge_index, self.gat_meta_1,   self.gat_meta_2)
        return self.semantic_attention([z1, z2])

    def predict(self, embeddings, lnc_ids, mir_ids, n_lnc):
        """
        Predict interaction scores for a batch of lncRNA–miRNA pairs.

        lnc_ids: tensor of lncRNA node indices (0-based within lncRNA type)
        mir_ids: tensor of miRNA node indices  (0-based within miRNA type)
        n_lnc:   number of lncRNA nodes (offset to reach miRNA in flat matrix)
        """
        lnc_ids = lnc_ids.to(embeddings.device)
        mir_ids = mir_ids.to(embeddings.device)
        lnc_emb = embeddings[lnc_ids]
        mir_emb = embeddings[mir_ids + n_lnc]
        pair    = torch.cat([lnc_emb, mir_emb], dim=1)
        return self.mlp(pair).squeeze(-1)
