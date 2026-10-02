import numpy as np

from adagenebudget import select_adagenebudget, select_top_expression
from adagenebudget.backends.scgpt import restore_native_order
from adagenebudget.backends.geneformer import reorder_subset_by_native_rank


def test_scgpt_stable_ties_then_native_order():
    # CSR/native positions arrive sorted. Stable score ties retain that order.
    positions = np.array([1, 4, 9, 12])
    values = np.array([2.0, 2.0, 1.0, 0.5])
    idf = np.ones(20)
    selected = select_adagenebudget(
        values, positions, idf, tau=0.60, k_min=1, k_max=3
    ).local_indices
    restored = restore_native_order(selected, positions)
    assert selected.tolist() == [0, 1]
    assert restored.tolist() == [0, 1]


def test_geneformer_token_id_tie_break_then_native_reorder():
    row_positions = np.array([0, 1, 2, 3])
    values = np.array([5.0, 5.0, 4.0, 1.0])
    token_ids = np.array([20, 10, 30, 40])
    top = select_top_expression(
        values, budget=2, tie_breaker=token_ids[row_positions]
    ).local_indices
    # Expression tie is broken by token ID ascending: local 1 before local 0.
    assert top.tolist() == [1, 0]

    gene_medians = np.array([5.0, 1.0, 4.0, 1.0])
    ordered = reorder_subset_by_native_rank(
        top,
        row_positions,
        values,
        gene_medians,
        token_ids=token_ids,
        n_counts=15.0,
    )
    # Geneformer native rank values are 5/1 > 5/5.
    assert ordered.tolist() == [1, 0]
