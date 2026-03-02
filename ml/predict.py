import logging
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_PATH = "models/spam_classifier_v2.pkl"

model_artifacts = joblib.load(MODEL_PATH)

pipeline = model_artifacts["pipeline"]
best_threshold = model_artifacts["threshold"]
classes = list(model_artifacts["classes"])


def predict_sms(text):
    X = pd.Series([text])
    
    spam_idx = classes.index("spam")
    spam_probability = pipeline.predict_proba(X)[0][spam_idx]
    
    if spam_probability >= best_threshold:
        return "spam"
    else:
        return "ham"


if __name__ == "__main__":
    logging.info(f"Model loaded. Using optimized threshold: {best_threshold:.4f}")
    while True:
        msg = input("\nEnter SMS (or Ctrl+C to exit): ")
        if not msg.strip():
            continue
        prediction = predict_sms(msg)
        logging.info(f"Prediction: {prediction}")
