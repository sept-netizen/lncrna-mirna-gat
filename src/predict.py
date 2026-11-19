# ================================================================
#  PREDICTION INTERFACE
#  Given a lncRNA name and miRNA name → interaction probability
# ================================================================

import torch


def predict_interaction(model, hetero_data, direct_edges,
                        metapath_edges, lnc_to_id, mir_to_id, n_lnc,
                        lncrna, mirna):
    """
    Predict interaction probability for one lncRNA–miRNA pair.

    Parameters:
        lncrna : str  e.g. "HOTTIP"
        mirna  : str  e.g. "hsa-miR-21-5p"

    Returns:
        probability (float), verdict (str)
    """
    if lncrna not in lnc_to_id:
        return None, f"lncRNA '{lncrna}' not in dataset."
    if mirna not in mir_to_id:
        return None, f"miRNA '{mirna}' not in dataset."

    lnc_id = torch.tensor([lnc_to_id[lncrna]])
    mir_id = torch.tensor([mir_to_id[mirna]])

    model.eval()
    with torch.no_grad():
        emb   = model.encode(hetero_data, direct_edges, metapath_edges)
        score = model.predict(emb, lnc_id, mir_id, n_lnc)
        prob  = torch.sigmoid(score).item()

    if prob >= 0.7:
        verdict = "✅ LIKELY INTERACT"
    elif prob >= 0.4:
        verdict = "⚠️  UNCERTAIN"
    else:
        verdict = "❌ UNLIKELY TO INTERACT"

    return prob, verdict