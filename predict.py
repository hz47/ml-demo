import pandas as pd
import joblib

# 1. Load the "toolbox" dictionary
model_artifacts = joblib.load("models/spam_classifier_v2.pkl")

# 2. Unpack the components
pipeline = model_artifacts["pipeline"]
best_threshold = model_artifacts["threshold"]
classes = list(model_artifacts["classes"])

def predict_sms(text):
    # Convert to pandas Series as the pipeline expects
    X = pd.Series([text])
    
    # 3. Use predict_proba to get the specific spam probability
    # We find the index of 'spam' to make sure we pull the right column
    spam_idx = classes.index("spam")
    spam_probability = pipeline.predict_proba(X)[0][spam_idx]
    
    # 4. Compare probability against our custom tuned threshold
    if spam_probability >= best_threshold:
        return "spam"
    else:
        return "ham"

if __name__ == "__main__":
    print(f"Model loaded. Using optimized threshold: {best_threshold:.4f}")
    while True:
        msg = input("\nEnter SMS (or Ctrl+C to exit): ")
        if not msg.strip():
            continue
        prediction = predict_sms(msg)
        print(f"Prediction: {prediction}")