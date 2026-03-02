import logging
import pandas as pd
import joblib
from config import MODELS_DIR
from data.clean import clean_text_v3 
from ml.utils import SMSMetadataExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load Model
MODEL_PATH = MODELS_DIR / "svm_spam_model.pkl"
model_artifacts = joblib.load(MODEL_PATH)

pipeline = model_artifacts["model"]
best_threshold = model_artifacts["threshold"]
classes = list(pipeline.classes_)

def predict_sms(text):
    # 2. TRANSFORM THE TEXT BEFORE PREDICTION
    cleaned_text = clean_text_v3(text)
    
    # 3. USE BOTH RAW AND CLEANED COLUMNS
    X = pd.DataFrame({
        'text': [text],           # Used by MetadataExtractor
        'clean_light': [cleaned_text] # Used by TF-IDF Vectorizer
    })
    
    spam_idx = classes.index("spam")
    # Using predict_proba to apply your custom threshold
    spam_probability = pipeline.predict_proba(X)[0][spam_idx]
    
    label = "spam" if spam_probability >= best_threshold else "ham"
    
    # Optional: Log the probability to see how close it was
    logging.info(f"Prob: {spam_probability:.4f} | Cleaned: {cleaned_text[:50]}...")
    
    return label

if __name__ == "__main__":
    logging.info(f"Model loaded. Using optimized threshold: {best_threshold:.4f}")
    try:
        while True:
            msg = input("\nEnter SMS (or Ctrl+C to exit): ")
            if not msg.strip():
                continue
            prediction = predict_sms(msg)
            logging.info(f"Prediction: {prediction}")
    except KeyboardInterrupt:
        print("\nExiting...")