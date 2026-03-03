import logging
import pandas as pd
import numpy as np
from config import SMS_CLEAN_PATH
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# class SMSMetadataExtractor(BaseEstimator, TransformerMixin):
#     """Common metadata extractor for LR/SVM/RF models"""

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         features = []
#         triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher']
#         for text in X:
#             text = str(text).lower()
#             char_count = len(text)
#             upper_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1)
#             trigger_count = sum(1 for t in triggers if t in text)
#             excl_count = text.count('!')
#             features.append([char_count, upper_ratio, trigger_count, excl_count])
#         return np.array(features)



class SMSMetadataExtractor(BaseEstimator, TransformerMixin):
    """Common metadata extractor for LR/SVM/RF models"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        triggers = ['win', 'won', 'prize', 'cash', 'claim', 'urgent', 'free', 'tone', 'voucher', 'cost', 'charge']
        
        currency_pattern = re.compile(r'£|\$|€|pound|pence', re.IGNORECASE)
        phone_pattern = re.compile(r'\b\d{10,13}\b')
        email_pattern = re.compile(r'subject:|from:|alertfrom|size:|adult 18', re.IGNORECASE)

        for text_raw in X:
            text_raw = str(text_raw)
            text_lower = text_raw.lower()
            
            char_count = len(text_raw)
            # uppercase calculation on the RAW text
            upper_ratio = sum(1 for c in text_raw if c.isupper()) / (char_count + 1)
            
            trigger_count = sum(1 for t in triggers if t in text_lower)
            excl_count = text_raw.count('!')
            
            # Explicit binary flags to bypass TF-IDF dilution
            has_currency = 1 if currency_pattern.search(text_raw) else 0
            has_long_num = 1 if phone_pattern.search(text_raw) else 0
            has_email_adult = 1 if email_pattern.search(text_raw) else 0
            
            features.append([
                char_count, 
                upper_ratio, 
                trigger_count, 
                excl_count,
                has_currency,
                has_long_num,
                has_email_adult
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



def find_best_f1_threshold(y_true, probs):
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precisions, recalls, thresholds = precision_recall_curve(y_true, probs, pos_label="spam")

    # Compute F2 score (beta=2)
    beta = 2
    f2_scores = (1 + beta**2) * (precisions[:-1] * recalls[:-1]) / ((beta**2 * precisions[:-1]) + recalls[:-1] + 1e-10)

    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx]