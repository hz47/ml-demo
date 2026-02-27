import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunAll:

    @patch("run_all.run_benchmark")
    @patch("run_all.run_preprocessing")
    @patch("run_all.run_word_analysis")
    @patch("run_all.run_ngrams")
    @patch("run_all.run_model_training")
    @patch("sys.argv", ["run_all.py"])
    def test_main_runs_all_steps(self, mock_train, mock_ngrams, mock_word, mock_clean, mock_benchmark):
        mock_word.return_value = ["word1", "word2"]
        mock_ngrams.return_value = (["bigram1"], ["trigram1"])
        
        from run_all import main
        main()
        
        mock_clean.assert_called_once()
        mock_word.assert_called_once()
        mock_ngrams.assert_called_once()
        mock_train.assert_called_once()
        mock_benchmark.assert_called_once()

    @patch("run_all.run_preprocessing")
    @patch("run_all.run_word_analysis")
    @patch("run_all.run_ngrams")
    @patch("run_all.run_model_training")
    @patch("sys.argv", ["run_all.py"])
    def test_main_handles_exception(self, mock_train, mock_ngrams, mock_word, mock_clean):
        mock_clean.side_effect = Exception("Preprocessing failed")
        
        from run_all import main
        import logging
        
        with patch("run_all.logging.exception") as mock_log:
            with pytest.raises(SystemExit):
                main()
            mock_log.assert_called_once()
