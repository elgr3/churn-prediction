from churn_prediction import config


def test_age_labels_match_bins():
    assert len(config.AGE_LABELS) == len(config.AGE_BINS) - 1
