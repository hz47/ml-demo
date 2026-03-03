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


def check_overfitting(pipeline, X, y, cv=5, scoring='f1_weighted'):
    """
    Check for overfitting by comparing train score vs cross-validation score.
    
    Args:
        pipeline: sklearn pipeline with fit method
        X: Features (DataFrame or array)
        y: Labels
        cv: Number of cross-validation folds
        scoring: Scoring metric ('f1_weighted', 'accuracy', 'precision_weighted', etc.)
    
    Returns:
        dict: {
            'train_score': float,
            'cv_score_mean': float,
            'cv_score_std': float,
            'overfitting_gap': float,
            'is_overfitting': bool
        }
    """
    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    
    pipeline_copy = pipeline.fit(X, y)
    train_score = scoring_to_func(scoring)(pipeline_copy, X, y)
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_score - cv_mean
    
    return {
        'train_score': train_score,
        'cv_score_mean': cv_mean,
        'cv_score_std': cv_std,
        'overfitting_gap': gap,
        'is_overfitting': gap > 0.05 or train_score >= 0.99
    }


def scoring_to_func(scoring):
    """Return appropriate scoring function."""
    from sklearn.metrics import get_scorer
    scorer = get_scorer(scoring)
    return scorer
