# ================================================================
#  DATA LOADER
#  Loads all 6 interaction files, assigns node IDs,
#  builds edge indices, constructs HeteroData graph,
#  attaches node features.
#
#  Memory notes (so the FULL dataset fits in Colab RAM):
#    - only the two needed columns of each CSV are read
#    - everything is vectorised (no df.iterrows())
#    - metapath edges are built with a sparse matrix product
#      instead of Python lists of every lncRNA×miRNA pair
# ================================================================

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
from config import (
    FILES, SEP,
    LM_LNCRNA_COL, LM_MIRNA_COL,
    LP_LNCRNA_COL, LP_PROTEIN_COL,
    MP_MIRNA_COL,  MP_PROTEIN_COL,
    PD_PROTEIN_COL, PD_DISEASE_COL,
    LD_LNCRNA_COL, LD_DISEASE_COL,
    MD_MIRNA_COL,  MD_DISEASE_COL,
    RNA_FEATURE_DIM, GENERIC_FEATURE_DIM,
    MAX_METAPATH_EDGES
)

# (source column, destination column) for every file
COLUMNS = {
    'lncrna_mirna'    : (LM_LNCRNA_COL,  LM_MIRNA_COL),
    'lncrna_protein'  : (LP_LNCRNA_COL,  LP_PROTEIN_COL),
    'mirna_protein'   : (MP_MIRNA_COL,   MP_PROTEIN_COL),
    'protein_disease' : (PD_PROTEIN_COL, PD_DISEASE_COL),
    'lncrna_disease'  : (LD_LNCRNA_COL,  LD_DISEASE_COL),
    'mirna_disease'   : (MD_MIRNA_COL,   MD_DISEASE_COL),
}


def load_file(key):
    """
    Load only the two needed columns of one interaction file.
    The first line of every file is a header row, so it is skipped
    (otherwise names like 'ncRNA_Symbol' become fake nodes).
    Returns a DataFrame with columns ['src', 'dst'] of clean strings.
    """
    src_col, dst_col = COLUMNS[key]
    df = pd.read_csv(
        FILES[key], sep=SEP, header=None, skiprows=1,
        usecols=[src_col, dst_col], dtype=str, on_bad_lines='skip'
    )
    df = df.rename(columns={src_col: 'src', dst_col: 'dst'})
    df = df.dropna()
    df['src'] = df['src'].str.strip()
    df['dst'] = df['dst'].str.strip()
    return df.drop_duplicates()


def build_node_mappings(dfs):
    """
    Collect all unique entities across all files.
    Returns four dicts mapping entity name → integer node ID.
    """
    lncrnas  = set()
    mirnas   = set()
    proteins = set()
    diseases = set()

    lncrnas.update( dfs['lncrna_mirna'   ]['src'])
    mirnas.update(  dfs['lncrna_mirna'   ]['dst'])

    lncrnas.update( dfs['lncrna_protein' ]['src'])
    proteins.update(dfs['lncrna_protein' ]['dst'])

    mirnas.update(  dfs['mirna_protein'  ]['src'])
    proteins.update(dfs['mirna_protein'  ]['dst'])

    proteins.update(dfs['protein_disease']['src'])
    diseases.update(dfs['protein_disease']['dst'])

    lncrnas.update( dfs['lncrna_disease' ]['src'])
    diseases.update(dfs['lncrna_disease' ]['dst'])

    mirnas.update(  dfs['mirna_disease'  ]['src'])
    diseases.update(dfs['mirna_disease'  ]['dst'])

    lnc_to_id = {n: i for i, n in enumerate(sorted(lncrnas))}
    mir_to_id = {n: i for i, n in enumerate(sorted(mirnas))}
    pro_to_id = {n: i for i, n in enumerate(sorted(proteins))}
    dis_to_id = {n: i for i, n in enumerate(sorted(diseases))}

    return lnc_to_id, mir_to_id, pro_to_id, dis_to_id


def map_ids(df, src_map, dst_map):
    """
    Convert a ['src', 'dst'] name DataFrame into two numpy arrays
    of integer IDs (local to each node type), dropping unknown names.
    """
    s = df['src'].map(src_map)
    d = df['dst'].map(dst_map)
    keep = s.notna() & d.notna()
    return s[keep].to_numpy(np.int64), d[keep].to_numpy(np.int64)


def make_edge_index(df, src_map, dst_map):
    """
    Build an edge index tensor [2, E] for one typed relation.
    IDs are local to each node type (standard HeteroData format).
    """
    s, d = map_ids(df, src_map, dst_map)
    return torch.from_numpy(np.stack([s, d]))


def build_hetero_graph(lnc_to_id, mir_to_id, pro_to_id, dis_to_id, dfs):
    """
    Construct PyG HeteroData object with all edge types.
    Node features are placeholders — attached separately by features.py.
    """
    data = HeteroData()

    # Placeholder features — overwritten by attach_features()
    data['lncrna' ].x = torch.zeros(len(lnc_to_id), RNA_FEATURE_DIM)
    data['mirna'  ].x = torch.zeros(len(mir_to_id),  RNA_FEATURE_DIM)
    data['protein'].x = torch.zeros(len(pro_to_id),  GENERIC_FEATURE_DIM)
    data['disease'].x = torch.zeros(len(dis_to_id),  GENERIC_FEATURE_DIM)

    data['lncrna',  'interacts',  'mirna'  ].edge_index = make_edge_index(
        dfs['lncrna_mirna'],    lnc_to_id, mir_to_id)
    data['lncrna',  'binds',      'protein'].edge_index = make_edge_index(
        dfs['lncrna_protein'],  lnc_to_id, pro_to_id)
    data['mirna',   'targets',    'protein'].edge_index = make_edge_index(
        dfs['mirna_protein'],   mir_to_id, pro_to_id)
    data['protein', 'associates', 'disease'].edge_index = make_edge_index(
        dfs['protein_disease'], pro_to_id, dis_to_id)
    data['lncrna',  'linked_to',  'disease'].edge_index = make_edge_index(
        dfs['lncrna_disease'],  lnc_to_id, dis_to_id)
    data['mirna',   'linked_to',  'disease'].edge_index = make_edge_index(
        dfs['mirna_disease'],   mir_to_id, dis_to_id)

    return data


def build_metapath_edges(lnc_to_id, mir_to_id, pro_to_id, dfs):
    """
    Explicitly enumerate lncRNA → Protein → miRNA metapath instances.

    A lncRNA and a miRNA are connected if they share at least one
    protein. Computed as a sparse matrix product:
        (lncRNA × protein) @ (protein × miRNA) = (lncRNA × miRNA)
    where each entry counts how many proteins the pair shares.

    Hub proteins (e.g. AGO2) can link almost every lncRNA to almost
    every miRNA, so if there are more than MAX_METAPATH_EDGES pairs
    we keep the ones supported by the MOST shared proteins
    (strongest evidence). No input rows are thrown away.

    Returns tensor [2, E] with local IDs (lncRNA id, miRNA id).
    """
    n_lnc, n_mir, n_pro = len(lnc_to_id), len(mir_to_id), len(pro_to_id)

    l, p = map_ids(dfs['lncrna_protein'], lnc_to_id, pro_to_id)
    m, q = map_ids(dfs['mirna_protein'],  mir_to_id, pro_to_id)

    A_lp = sp.csr_matrix((np.ones(len(l), np.float32), (l, p)), shape=(n_lnc, n_pro))
    A_pm = sp.csr_matrix((np.ones(len(m), np.float32), (q, m)), shape=(n_pro, n_mir))
    A_lp.data[:] = 1   # duplicates were summed → reset to binary
    A_pm.data[:] = 1

    C = (A_lp @ A_pm).tocoo()   # C[i, j] = #proteins shared by lnc i and mir j

    if C.nnz == 0:
        print("Warning: no lncRNA→Protein→miRNA metapath instances found.")
        return torch.zeros((2, 0), dtype=torch.long)

    rows, cols, counts = C.row, C.col, C.data
    print(f"  Metapath pairs found (lncRNA→Protein→miRNA): {C.nnz}")

    if C.nnz > MAX_METAPATH_EDGES:
        top = np.argpartition(-counts, MAX_METAPATH_EDGES)[:MAX_METAPATH_EDGES]
        rows, cols = rows[top], cols[top]
        print(f"  Kept the {MAX_METAPATH_EDGES} strongest "
              f"(most shared proteins) to fit in memory")

    return torch.from_numpy(np.stack([rows.astype(np.int64),
                                      cols.astype(np.int64)]))


def to_flat_edges(lnc_mir_edges, n_lnc):
    """
    The GAT layers work on ONE flat node matrix ordered
    [lncRNA | miRNA | protein | disease], so a miRNA with local id j
    sits at row n_lnc + j. Shift miRNA ids by n_lnc and add the
    reverse direction so messages flow both ways.

    lnc_mir_edges : tensor [2, E] of local (lncRNA id, miRNA id)
    Returns       : tensor [2, 2E] in flat-matrix indices
    """
    lnc = lnc_mir_edges[0]
    mir = lnc_mir_edges[1] + n_lnc
    return torch.stack([torch.cat([lnc, mir]), torch.cat([mir, lnc])])


def load_all():
    """
    Master loader — runs the full data pipeline and returns
    everything needed by the rest of the project.
    """
    print("Loading interaction files...")
    dfs = {key: load_file(key) for key in FILES}

    print("Building node mappings...")
    lnc_to_id, mir_to_id, pro_to_id, dis_to_id = build_node_mappings(dfs)
    print(f"  lncRNAs : {len(lnc_to_id)}")
    print(f"  miRNAs  : {len(mir_to_id)}")
    print(f"  Proteins: {len(pro_to_id)}")
    print(f"  Diseases: {len(dis_to_id)}")

    print("Building HeteroData graph...")
    hetero_data = build_hetero_graph(
        lnc_to_id, mir_to_id, pro_to_id, dis_to_id, dfs
    )

    print("Building metapath edges...")
    metapath_edges = build_metapath_edges(
        lnc_to_id, mir_to_id, pro_to_id, dfs
    )

    print("Attaching node features...")
    from features import attach_features
    hetero_data = attach_features(
        hetero_data,
        lnc_to_id, mir_to_id, pro_to_id, dis_to_id
    )

    print("Extracting positive lncRNA–miRNA pairs...")
    ei = hetero_data['lncrna', 'interacts', 'mirna'].edge_index
    pos_pairs = list(set(zip(ei[0].tolist(), ei[1].tolist())))
    print(f"  Positive pairs: {len(pos_pairs)}")

    return (hetero_data, metapath_edges, pos_pairs,
            lnc_to_id, mir_to_id, pro_to_id, dis_to_id)
