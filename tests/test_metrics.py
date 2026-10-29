import math

import pytest

from eval import metrics


def test_recall_mrr_ndcg_hand_computed():
    ranked = [7, 3, 9, 1, 5]
    gold = {3, 5}
    assert metrics.recall_at_k(ranked, gold, 1) == 0.0
    assert metrics.recall_at_k(ranked, gold, 2) == 0.5
    assert metrics.recall_at_k(ranked, gold, 5) == 1.0
    assert metrics.reciprocal_rank(ranked, gold) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(expected)


def test_not_found():
    assert metrics.reciprocal_rank([1, 2], {9}) == 0.0
    assert metrics.recall_at_k([1, 2], {9}, 10) == 0.0


def test_empty_gold_is_error():
    with pytest.raises(ValueError):
        metrics.recall_at_k([1], set(), 1)


def test_bootstrap_detects_clear_improvement():
    a = [0.0] * 50 + [1.0] * 50
    b = [1.0] * 100
    res = metrics.paired_bootstrap(a, b, n_boot=2000)
    assert res["delta"] == 0.5 and res["p_value"] < 0.01


def test_cluster_bootstrap_point_estimates_hand_computed():
    # cụm A: 3 truy vấn, mỗi truy vấn tăng 1; cụm B: 1 truy vấn không đổi
    a = [0.0, 0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0, 0.0]
    clusters = ["A", "A", "A", "B"]
    q = metrics.paired_cluster_bootstrap(a, b, clusters, n_boot=2000, estimand="query")
    c = metrics.paired_cluster_bootstrap(a, b, clusters, n_boot=2000, estimand="cluster")
    assert q["delta"] == pytest.approx(0.75)        # trung bình theo truy vấn
    assert c["delta"] == pytest.approx(0.5)         # trung bình của trung bình từng cụm
    assert q["n_clusters"] == 2


def test_cluster_bootstrap_wider_than_query_bootstrap_for_correlated_queries():
    # 20 cụm, mỗi cụm 10 truy vấn cùng một chênh lệch: các truy vấn trong cụm không độc lập
    import numpy as np
    rng = np.random.default_rng(0)
    effects = rng.normal(0.05, 0.2, 20)
    d = np.repeat(effects, 10)
    clusters = np.repeat(np.arange(20), 10)
    plain = metrics.paired_bootstrap(np.zeros_like(d), d)
    clus = metrics.paired_cluster_bootstrap(np.zeros_like(d), d, clusters)
    assert clus["ci95_high"] - clus["ci95_low"] > 2 * (plain["ci95_high"] - plain["ci95_low"])


def test_permutation_identical_systems_give_p_one():
    # hai hệ thống giống hệt nhau: mọi phép hoán vị cho cùng thống kê nên p = 1 (bootstrap cũ cho p hai phía = 0)
    for est in ("query", "cluster"):
        r = metrics.paired_cluster_permutation([0.0, 0.0], [0.0, 0.0], ["A", "B"], estimand=est)
        assert r["delta"] == 0.0 and r["p"] == 1.0


def test_permutation_ties_at_zero():
    # cụm A giảm 1, cụm B không đổi: hai phép đổi dấu của A cho |T| như nhau nên p hai phía = 1
    r = metrics.paired_cluster_permutation([1.0, 1.0], [0.0, 1.0], ["A", "B"])
    assert r["delta"] == pytest.approx(-0.5) and r["p"] == 1.0


def test_permutation_exact_enumeration_hand_computed():
    # ba cụm một truy vấn, chênh lệch 1, 2, 3: tổng có dấu thuộc {±6, ±4, ±2, 0, 0};
    # |tổng| >= 6 ở 2 trên 8 phép đổi dấu, tổng >= 6 ở 1 trên 8
    a, b = [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]
    two = metrics.paired_cluster_permutation(a, b, ["A", "B", "C"])
    one = metrics.paired_cluster_permutation(a, b, ["A", "B", "C"], alternative="greater")
    assert two["exact"] and two["n_perm"] == 8 and two["p"] == pytest.approx(0.25)
    assert one["p"] == pytest.approx(0.125)


def test_permutation_two_sided_is_symmetric_and_monte_carlo_matches_exact():
    import numpy as np
    rng = np.random.default_rng(1)
    a = rng.random(60)
    b = a + rng.normal(0.08, 0.3, 60)
    clusters = np.repeat(np.arange(15), 4)          # 15 cụm, mỗi cụm 4 truy vấn
    exact = metrics.paired_cluster_permutation(a, b, clusters)
    swapped = metrics.paired_cluster_permutation(b, a, clusters)
    mc = metrics.paired_cluster_permutation(a, b, clusters, exact_max=0, n_perm=20000)
    assert exact["exact"] and not mc["exact"]
    assert swapped["p"] == pytest.approx(exact["p"])
    assert mc["p"] == pytest.approx(exact["p"], abs=0.01)


def test_permutation_cluster_estimand_weights_clusters_equally():
    # cụm A: 3 truy vấn tăng 1; cụm B: 1 truy vấn giảm 3. Theo truy vấn tổng bằng 0, theo cụm là (1 - 3) / 2 = -1.
    # Theo cụm, bốn phép đổi dấu cho (±1 ± 3) / 2 thuộc {-1, -2, 2, 1}, đều có |T| >= 1 nên p = 1
    a = [0.0, 0.0, 0.0, 3.0]
    b = [1.0, 1.0, 1.0, 0.0]
    q = metrics.paired_cluster_permutation(a, b, ["A", "A", "A", "B"], estimand="query")
    c = metrics.paired_cluster_permutation(a, b, ["A", "A", "A", "B"], estimand="cluster")
    assert q["delta"] == pytest.approx(0.0) and q["p"] == 1.0
    assert c["delta"] == pytest.approx(-1.0) and c["p"] == 1.0 and c["n_perm"] == 4


def test_holm_hand_computed():
    assert metrics.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
