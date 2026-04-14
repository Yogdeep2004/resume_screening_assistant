import re
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class MRPipeline:
    def __init__(self):
        self.tfidf = None
        self.svm_model = None
        self.encoder = None
        self.load_models()

    def load_models(self):
        try:
            self.tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
            self.svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
            self.encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
            print("✅ Successfully loaded pre-trained models.")
        except Exception as e:
            print(f"⚠️  Warning: Could not load models - {e}")

    def clean_text(self, text: str) -> str:
        text = re.sub(r'http\S+\s*', ' ', text)         # remove URLs
        text = re.sub(r'RT|cc', ' ', text)               # remove RT and cc
        text = re.sub(r'#\S+', '', text)                 # remove hashtags
        text = re.sub(r'@\S+', '  ', text)              # remove mentions
        text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)  # punctuations
        text = re.sub(r'[^\x00-\x7f]', r' ', text)      # non-ASCII
        text = re.sub(r'\s+', ' ', text)                 # extra whitespace
        return text.strip().lower()

    def predict_category(self, raw_text: str) -> str:
        if not self.svm_model or not self.tfidf or not self.encoder:
            return "Unknown (Models Missing)"
        cleaned = self.clean_text(raw_text)
        vectorized = self.tfidf.transform([cleaned])
        prediction = self.svm_model.predict(vectorized)
        category = self.encoder.inverse_transform(prediction)
        return category[0]


# Singleton instance
pipeline = MRPipeline()
