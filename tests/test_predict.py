import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.predict import predict_sms


class TestPredict:
    @pytest.fixture
    def model_exists(self):
        model_path = "models/spam_classifier_v2.pkl"
        if not os.path.exists(model_path):
            pytest.skip("Model not found, run training first")

    def test_predict_returns_string(self, model_exists):
        result = predict_sms("Hello, how are you?")
        assert isinstance(result, str)
        assert result in ["ham", "spam"]

    def test_predict_spam_keywords(self, model_exists):
        result = predict_sms("WIN FREE MONEY NOW!!!")
        assert result == "spam"

    def test_predict_ham_message(self, model_exists):
        result = predict_sms("Hey, are we still meeting for lunch?")
        assert result == "ham"
