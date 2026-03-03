import logging
import pandas as pd
import numpy as np
from config import SMS_CLEAN_PATH
from sklearn.metrics import precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class SMSMetadataExtractor(BaseEstimator, TransformerMixin):
    """Common metadata extractor for LR/SVM/RF models"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher']
        for text in X:
            text = str(text).lower()
            char_count = len(text)
            upper_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1)
            trigger_count = sum(1 for t in triggers if t in text)
            excl_count = text.count('!')
            features.append([char_count, upper_ratio, trigger_count, excl_count])
        return np.array(features)


class SMSMetadataExtractorNB(BaseEstimator, TransformerMixin):
    """Metadata extractor for Naive Bayes model"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher']
        for text in X:
            text = str(text)
            char_count = len(text)
            words = text.split()
            word_count = len(words) + 1

            avg_word_len = char_count / word_count
            sent_count = text.count('.') + text.count('!') + text.count('?')
            trigger_count = sum(1 for t in triggers if t in text.lower())

            features.append([
                char_count,
                avg_word_len,
                sent_count,
                trigger_count
            ])
        return np.array(features)


def load_data():
    logging.info(f"Loading data from {SMS_CLEAN_PATH}")
    df = pd.read_csv(SMS_CLEAN_PATH).dropna()
    logging.info(f"Loaded {len(df)} records")
    return df


def find_high_precision_threshold(y_true, probs, target_precision=1.0, default=0.98):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs, pos_label="spam")
    valid_idx = np.where(precisions >= target_precision)[0]
    return thresholds[valid_idx[0]] if len(valid_idx) > 0 else default
