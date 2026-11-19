# ================================================================
#  CONFIG — ALL COLUMN NAMES, FILE PATHS, HYPERPARAMETERS
#  If your column positions differ, change ONLY this file.
# ================================================================

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── File paths ───────────────────────────────────────────────────
FILES = {
    'lncrna_mirna'    : os.path.join(DATA_DIR, 'lncrna_mirna.csv'),
    'lncrna_protein'  : os.path.join(DATA_DIR, 'lncrna_protein.csv'),
    'mirna_protein'   : os.path.join(DATA_DIR, 'mirna_protein.csv'),
    'protein_disease' : os.path.join(DATA_DIR, 'protein_disease.csv'),
    'lncrna_disease'  : os.path.join(DATA_DIR, 'lncrna_disease.csv'),
    'mirna_disease'   : os.path.join(DATA_DIR, 'mirna_disease.csv'),
}

# ── Separator ────────────────────────────────────────────────────
# '\t' for tab-separated, ',' for comma-separated
SEP = ','

# ── lncrna_mirna columns ─────────────────────────────────────────
LM_LNCRNA_COL = 1    # lncRNA name
LM_MIRNA_COL  = 4    # miRNA name

# ── lncrna_protein columns ───────────────────────────────────────
LP_LNCRNA_COL  = 1   # lncRNA name
LP_PROTEIN_COL = 4   # protein name

# ── mirna_protein columns ────────────────────────────────────────
MP_MIRNA_COL   = 2   # miRNA name
MP_PROTEIN_COL = 4   # protein name

# ── protein_disease columns ──────────────────────────────────────
PD_PROTEIN_COL = 2   # protein name
PD_DISEASE_COL = 4   # disease name

# ── lncrna_disease columns ───────────────────────────────────────
LD_LNCRNA_COL  = 1   # lncRNA name
LD_DISEASE_COL = 4   # disease name

# ── mirna_disease columns ────────────────────────────────────────
MD_MIRNA_COL   = 1   # miRNA name
MD_DISEASE_COL = 2   # disease name

# ── Feature dimensions ───────────────────────────────────────────
RNA_FEATURE_DIM     = 68   # lncRNA and miRNA: composition (AUGC) + 3-mer
GENERIC_FEATURE_DIM = 16   # protein and disease: random init, model learns to adjust

# ── Model hyperparameters ────────────────────────────────────────
HIDDEN_DIM  = 128 #size of model's internal representation of each node
GAT_HEADS   = 8    # layer 1 uses HIDDEN_DIM // GAT_HEADS dims per head
DROPOUT     = 0.6
MLP_DROPOUT = 0.5

# ── Training hyperparameters ─────────────────────────────────────
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 200
LEARNING_RATE   = 0.001
NEG_RATIO       = 5
TRAIN_SPLIT     = 0.8 #80% training and 20% testing
SEED            = 42

# ── Memory limits (these do NOT delete any data) ─────────────────
# Max lncRNA–miRNA metapath edges; strongest (most shared proteins) kept
MAX_METAPATH_EDGES    = 1_000_000
# Nodes sampled per pre-training epoch for the contrastive loss
# (the loss builds a [2*S, 2*S] matrix, so it must not use all nodes)
PRETRAIN_SAMPLE_NODES = 2048
# Max positive pairs used per fine-tuning epoch (a fresh random
# subset every epoch, so all pairs still get used over training)
MAX_POS_PER_EPOCH     = 50_000