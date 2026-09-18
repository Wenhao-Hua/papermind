from trainer.negative_ablation import select_negatives


def test_no_positive_leakage_and_equal_counts():
    for method in ("random", "bm25", "dense", "mixed"):
        result = select_negatives(method, {1, 3}, [1, 2, 3, 0, 4], [3, 4, 1, 0, 2], 3, 42)
        assert len(result) == 3
        assert set(result) == {0, 2, 4}


def test_strategies_change_order():
    assert select_negatives("bm25", {0}, [0, 1, 2], [2, 1, 0], 1, 42) == [1]
    assert select_negatives("dense", {0}, [0, 1, 2], [2, 1, 0], 1, 42) == [2]
