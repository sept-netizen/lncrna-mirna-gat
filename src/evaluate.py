# ================================================================
#  EVALUATION — ROC-AUC, PR-AUC, plots
# ================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)
from train import sample_negatives


def evaluate(model, hetero_data, direct_edges, metapath_edges,
             test_pos, pos_codes, n_lnc, n_mir):

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(hetero_data, direct_edges, metapath_edges)

        test_pos  = np.asarray(test_pos, dtype=np.int64)
        test_neg  = sample_negatives(pos_codes, n_lnc, n_mir, len(test_pos))
        all_pairs = torch.from_numpy(np.concatenate([test_pos, test_neg]))
        labels    = [1]*len(test_pos) + [0]*len(test_neg)

        scores = torch.sigmoid(
            model.predict(embeddings, all_pairs[:, 0], all_pairs[:, 1], n_lnc)
        ).cpu().numpy()

    roc_auc = roc_auc_score(labels, scores)
    pr_auc  = average_precision_score(labels, scores)

    print("── Results ───────────────────────────────────────")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # Plot
    fpr,  tpr,  _ = roc_curve(labels, scores)
    prec, rec,  _ = precision_recall_curve(labels, scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color='purple', lw=2,
             label=f'ROC-AUC = {roc_auc:.4f}')
    ax1.plot([0,1],[0,1],'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve — MetapathGAT')
    ax1.legend()

    ax2.plot(rec, prec, color='teal', lw=2,
             label=f'PR-AUC = {pr_auc:.4f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision–Recall Curve — MetapathGAT')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("  Plot saved → results.png")

    return roc_auc, pr_auc