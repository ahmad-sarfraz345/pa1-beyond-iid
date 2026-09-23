import numpy as np


def validation_threshold(validation_scores):
    """Accept scores at or below the validation-only 95th percentile."""
    return float(np.percentile(validation_scores, 95))
