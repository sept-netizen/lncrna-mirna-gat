# ================================================================
#  MAIN — runs the full pipeline end to end
#  Run with: python main.py
# ================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from data_loader import load_all, to_flat_edges
from model       import MetapathGAT
from pretrain    import pretrain
from train       import train, pair_codes
from evaluate    import evaluate
from predict     import predict_interaction
from config      import TRAIN_SPLIT, SEED

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Use the GPU if there is one (Colab: Runtime → Change runtime type → T4 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Step 1: Load data and build graph ─────────────────────
    (hetero_data, metapath_edges, pos_pairs,
     lnc_to_id, mir_to_id, pro_to_id, dis_to_id) = load_all()

    n_lnc     = len(lnc_to_id)
    n_mir     = len(mir_to_id)
    n_pro     = len(pro_to_id)
    n_dis     = len(dis_to_id)
    pos_codes = np.sort(pair_codes(pos_pairs, n_mir))   # all known positives

    # ── Step 2: Train/test split BEFORE building the graph ────
    # Test interactions must not be graph edges during training,
    # otherwise the model just looks the answer up (data leakage).
    train_pos, test_pos = train_test_split(
        pos_pairs, test_size=1-TRAIN_SPLIT, random_state=SEED
    )
    train_edges = torch.tensor(train_pos, dtype=torch.long).T   # [2, E]

    # Flat-index edges for the GAT layers (miRNA ids shifted by n_lnc)
    direct_edges   = to_flat_edges(train_edges,    n_lnc).to(device)
    metapath_edges = to_flat_edges(metapath_edges, n_lnc).to(device)
    hetero_data    = hetero_data.to(device)

    # ── Step 3: Build MetapathGAT ─────────────────────────────
    model = MetapathGAT(n_lnc, n_mir, n_pro, n_dis).to(device)

    # ── Step 4: Contrastive pre-training ──────────────────────
    model = pretrain(model, hetero_data, direct_edges)

    # ── Step 5: Fine-tune ──────────────────────────────────────
    model = train(
        model, hetero_data, direct_edges, metapath_edges,
        train_pos, pos_codes, n_lnc, n_mir
    )

    # ── Step 6: Evaluate ───────────────────────────────────────
    evaluate(
        model, hetero_data, direct_edges, metapath_edges,
        test_pos, pos_codes, n_lnc, n_mir
    )

    # ── Step 7: Interactive prediction ────────────────────────
    print("\n── Interactive Prediction ────────────────────────")
    print("Enter lncRNA and miRNA names to get interaction probability.")
    print("(Ctrl+C, or the Colab stop button, to quit)\n")

    while True:
        try:
            lnc = input("lncRNA : ").strip()
            mir = input("miRNA  : ").strip()
            prob, verdict = predict_interaction(
                model, hetero_data, direct_edges, metapath_edges,
                lnc_to_id, mir_to_id, n_lnc, lnc, mir
            )
            if prob is not None:
                print(f"→ Probability: {prob:.4f}  |  {verdict}\n")
            else:
                print(f"→ {verdict}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
