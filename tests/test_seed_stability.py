from evaluation.seed_stability import paired_paper_interval, percentile


def test_paired_cluster_resampling_preserves_constant_effect():
    assert paired_paper_interval([.2, .2, .2], [[0, 1], [2]], draws=30) == [.2, .2]


def test_percentile_interpolation():
    assert percentile([0, 1, 2, 3], .5) == 1.5
