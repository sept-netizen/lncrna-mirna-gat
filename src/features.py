# ================================================================
#  FEATURE ENGINEERING
#  Generates numerical feature vectors for RNA nodes.
#
#  For lncRNA and miRNA nodes: composition + k-mer features
#  For protein and disease nodes: identity-based features
#  (since we don't have sequences for proteins/diseases,
#   they get learned embeddings initialized from their node index)
# ================================================================

import numpy as np
import torch
from itertools import product
from config import GENERIC_FEATURE_DIM as FEATURE_DIM


# ── RNA sequence features ─────────────────────────────────────────

def composition_features(seq):
    """
    Fractional nucleotide composition.
    Returns 4-dim vector: [frac_A, frac_U, frac_C, frac_G]
    """
    seq = seq.upper().replace('T', 'U')  # normalize DNA to RNA
    length = max(len(seq), 1)
    return [seq.count(n) / length for n in 'AUCG']


def kmer_features(seq, k=3):
    """
    Normalized k-mer frequency vector.
    k=3 → 4^3 = 64 possible 3-mers → 64-dim vector
    """
    seq      = seq.upper().replace('T', 'U')
    alphabet = 'AUCG'
    kmers    = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}

    counts = [0] * len(kmers)
    total  = max(len(seq) - k + 1, 1)

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_idx:
            counts[kmer_idx[kmer]] += 1

    return [c / total for c in counts]


def rna_features(seq):
    """
    Full RNA feature vector: composition (4) + 3-mer (64) = 68-dim.
    If sequence is unavailable, returns a zero vector.
    """
    if not seq or len(seq) < 3:
        return [0.0] * 68
    comp = composition_features(seq)
    kmer = kmer_features(seq, k=3)
    return comp + kmer


# ── Node feature matrix builders ──────────────────────────────────

def build_rna_features(name_to_seq, id_map):
    """
    Build feature matrix for lncRNA or miRNA nodes.

    Parameters:
        name_to_seq : dict mapping node name → sequence string
                      (can be empty dict if sequences unavailable)
        id_map      : dict mapping node name → integer ID

    Returns:
        torch.FloatTensor of shape [N_nodes, 68]
        Falls back to random init if sequence unavailable.
    """
    n = len(id_map)
    matrix = np.zeros((n, 68), dtype=np.float32)

    for name, idx in id_map.items():
        seq = name_to_seq.get(name, '')
        matrix[idx] = rna_features(seq)

    # For nodes with no sequence (all zeros), replace with small random
    # so they're not all identical in embedding space
    zero_rows = np.where(matrix.sum(axis=1) == 0)[0]
    if len(zero_rows) > 0:
        matrix[zero_rows] = np.random.randn(
            len(zero_rows), 68
        ).astype(np.float32) * 0.01

    return torch.tensor(matrix, dtype=torch.float)


def build_generic_features(id_map, dim=FEATURE_DIM):
    """
    Build feature matrix for protein or disease nodes.
    Since we don't have sequences for these, we use small random
    vectors — the type-specific projection layer in the model
    (TypeProjection) will learn meaningful transformations from these.

    Returns:
        torch.FloatTensor of shape [N_nodes, dim]
    """
    n = len(id_map)
    return torch.randn(n, dim) * 0.01


def attach_features(hetero_data, lnc_to_id, mir_to_id,
                    pro_to_id, dis_to_id,
                    lnc_seqs=None, mir_seqs=None):
    """
    Attach feature matrices to the HeteroData graph object.

    Parameters:
        hetero_data : PyG HeteroData object
        lnc_to_id   : lncRNA name → node ID
        mir_to_id   : miRNA name  → node ID
        pro_to_id   : protein name → node ID
        dis_to_id   : disease name → node ID
        lnc_seqs    : dict {lncRNA_name: sequence} or None
        mir_seqs    : dict {miRNA_name:  sequence} or None

    If sequences are not provided, RNA nodes also get small
    random features (same as protein/disease nodes).
    The model still learns from graph structure via GAT.
    """
    lnc_seqs = lnc_seqs or {}
    mir_seqs = mir_seqs or {}

    # lncRNA and miRNA: use sequence features if available
    # otherwise fall back to random (handled inside build_rna_features)
    lnc_feat = build_rna_features(lnc_seqs, lnc_to_id)  # [N_lnc, 68]
    mir_feat = build_rna_features(mir_seqs, mir_to_id)  # [N_mir, 68]

    # Protein and disease: random init, learned by TypeProjection
    pro_feat = build_generic_features(pro_to_id, dim=FEATURE_DIM)
    dis_feat = build_generic_features(dis_to_id, dim=FEATURE_DIM)

    # Attach to graph
    hetero_data['lncrna' ].x = lnc_feat
    hetero_data['mirna'  ].x = mir_feat
    hetero_data['protein'].x = pro_feat
    hetero_data['disease'].x = dis_feat

    print(f"  lncRNA  features: {lnc_feat.shape}")
    print(f"  miRNA   features: {mir_feat.shape}")
    print(f"  Protein features: {pro_feat.shape}")
    print(f"  Disease features: {dis_feat.shape}")

    return hetero_data