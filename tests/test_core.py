import numpy as np
from scipy import sparse

from adagenebudget import compute_idf, match_fixed_budget, select_adagenebudget, select_top_expression
from adagenebudget.backends.scgpt import restore_native_order
from adagenebudget.backends.geneformer import reorder_subset_by_native_rank


def test_idf_matches_paper_formula():
    x = sparse.csr_matrix(np.array([[1,0,2],[0,0,3],[1,4,0]], dtype=float))
    got = compute_idf(x, dtype=np.float64)
    df = np.array([2,1,2], dtype=float)
    expected = np.log((3 + 1) / (df + 1)) + 1
    np.testing.assert_allclose(got, expected)


def test_adaptive_shortest_prefix_and_bounds():
    values = np.array([8., 1., 1.])
    positions = np.arange(3)
    idf = np.ones(3)
    r = select_adagenebudget(values, positions, idf, tau=.8, k_min=1, k_max=3)
    assert r.raw_k == 1 and r.selected_k == 1
    assert r.local_indices.tolist() == [0]

    r2 = select_adagenebudget(values, positions, idf, tau=.8, k_min=2, k_max=3)
    assert r2.raw_k == 1 and r2.selected_k == 2 and r2.lower_clipped


def test_scgpt_restore_native_order():
    positions = np.array([2, 5, 9, 12])
    selected = np.array([3, 0, 2])
    assert restore_native_order(selected, positions).tolist() == [0, 2, 3]


def test_geneformer_native_reorder():
    positions = np.array([0,1,2])
    values = np.array([10., 9., 8.])
    medians = np.array([10., 3., 8.])
    selected = np.array([0,1,2])
    got = reorder_subset_by_native_rank(selected, positions, values, medians)
    assert got.tolist() == [1, 0, 2]


def test_match_fixed_budget():
    counts = np.array([2,5,8,10])
    k, mean = match_fixed_budget(counts, target_mean=4.0)
    assert k > 0
    assert mean == np.minimum(counts, k).mean()
