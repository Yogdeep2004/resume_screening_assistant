import os
import sys
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

# Dummy dataset for sample training
sample_data = {
    'Resume': [
        "Experienced software engineer with expertise in Python, Java, React. Built scalable web applications and REST APIs. Git Jenkins CI/CD.",
        "Data scientist passionate about machine learning, deep learning, Python, TensorFlow, and natural language processing. Statistics modeling.",
        "HR professional skilled in recruitment, employee relations, talent acquisition, benefits administration, and HR policies compliance.",
        "Creative UI/UX designer with 5 years of experience in Figma, Adobe XD, Photoshop and user-centered design research.",
        "DevOps engineer specializing in AWS, Docker, Kubernetes, CI/CD pipelines and Terraform infrastructure as code.",
        "Full stack web developer proficient in React, Node.js, MongoDB, Express, building scalable SaaS products and RESTful APIs.",
        "Data analyst experienced in SQL, Excel, Power BI, Tableau for business intelligence and reporting and dashboarding.",
        "Cybersecurity analyst specializing in penetration testing, SIEM tools, vulnerability assessment, and incident response.",
    ],
    'Category': [
        "Software Engineering",
        "Data Science",
        "Human Resources",
        "Design",
        "DevOps",
        "Web Development",
        "Data Analytics",
        "Cybersecurity",
    ]
}

def clean_text(text: str) -> str:
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def train_and_save():
    print("🚀 Training sample models...")
    df = pd.DataFrame(sample_data)
    df['cleaned'] = df['Resume'].apply(clean_text)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['cleaned'])

    # Label Encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Category'])

    # SVM Model
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X, y)

    # Save to models directory (relative to this script's location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(tfidf, os.path.join(model_dir, "tfidf.pkl"))
    joblib.dump(svm_model, os.path.join(model_dir, "svm_model.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))

    print(f"✅ Models saved to {model_dir}/")

if __name__ == "__main__":
    train_and_save()
