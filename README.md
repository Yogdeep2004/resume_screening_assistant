# resume_screening_assistant


Here's the cleaned-up, professional version of the README with all emojis removed:

---

<div align="center">

# Resume Screening Assistant

### An AI-Powered Resume Classification & Screening Tool

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-154F3C?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <strong>Automate resume screening with Machine Learning and NLP. Upload a resume and instantly predict the job category it belongs to — saving recruiters hours of manual effort.</strong>
</p>

---

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture and How It Works](#architecture-and-how-it-works)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

The **Resume Screening Assistant** is an intelligent, end-to-end machine learning application that automates the tedious process of resume screening. In today's competitive job market, recruiters often sift through hundreds — sometimes thousands — of resumes for a single position. This tool leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** to instantly classify resumes into their most relevant job categories.

Simply upload a resume (as a `.txt` or `.pdf` file), and the model will predict the professional category (e.g., *Data Science, Java Developer, HR, Web Designing, etc.*) within seconds.

---

## Features

| Feature | Description |
|---|---|
| **Automated Resume Classification** | Upload a resume and get instant job category predictions powered by ML |
| **Interactive Web Interface** | Clean, user-friendly Streamlit dashboard for seamless interaction |
| **Text Preprocessing Pipeline** | Robust NLP pipeline including tokenization, stopword removal, lemmatization, and regex-based cleaning |
| **Trained ML Model** | Pre-trained classification model using TF-IDF vectorization and supervised learning algorithms |
| **Multi-Format Support** | Supports `.txt` and `.pdf` resume uploads |
| **Real-Time Predictions** | Near-instant predictions with pre-loaded model and vectorizer |
| **Multi-Category Support** | Classifies resumes across 25+ job categories |
| **Serialized Model** | Pre-trained model and vectorizer saved as pickle files for fast loading |

---

## Architecture and How It Works

```
+-------------------------------------------------------------+
|                    USER INTERFACE (Streamlit)                |
|                  Upload Resume (.pdf / .txt)                 |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  TEXT EXTRACTION LAYER                       |
|          Extract raw text from uploaded resume               |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               TEXT PREPROCESSING PIPELINE                    |
|  +-----------+  +----------+  +------------+  +-----------+ |
|  | Lowercase |->|  Regex   |->|  Stopword  |->|  Lemma-   | |
|  | Conversion|  | Cleaning |  |  Removal   |  | tization  | |
|  +-----------+  +----------+  +------------+  +-----------+ |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                TF-IDF VECTORIZATION                          |
|         Convert cleaned text to numerical features           |
|              (Pre-fitted TfidfVectorizer)                    |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              ML CLASSIFICATION MODEL                         |
|     Predict job category from feature vector                 |
|          (Pre-trained classifier via pickle)                 |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  PREDICTION OUTPUT                           |
|        Display predicted job category on dashboard           |
+-------------------------------------------------------------+
```

### Step-by-Step Workflow

1. **Upload** — User uploads a resume via the Streamlit web app.
2. **Extract** — Raw text is extracted from the uploaded file.
3. **Clean** — The text is preprocessed using NLP techniques:
   - Convert to lowercase
   - Remove URLs, special characters, and extra whitespace
   - Remove stopwords (common English words that add no value)
   - Apply lemmatization to reduce words to their base form
4. **Vectorize** — Cleaned text is transformed into a TF-IDF feature vector using the pre-fitted vectorizer.
5. **Predict** — The trained ML model classifies the feature vector into a job category.
6. **Display** — The predicted category is displayed on the Streamlit dashboard.

---

## Project Structure

```
resume_screening_assistant/
|
|-- resume_screening.ipynb     # Jupyter notebook for EDA, model training, and evaluation
|-- app.py                     # Streamlit web application (main entry point)
|-- cleantext.py               # Text preprocessing and cleaning utilities
|-- clf.pkl                    # Serialized trained classification model (pickle)
|-- tfidf.pkl                  # Serialized TF-IDF vectorizer (pickle)
|-- UpdatedResumeDataSet.csv   # Dataset used for training the model
|-- requirements.txt           # Python dependencies
|-- README.md                  # Project documentation (this file)
```

### File Descriptions

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application. Handles file uploads, invokes preprocessing, loads the model and vectorizer, and displays predictions. |
| `cleantext.py` | Contains the `cleanResume()` function that performs all text cleaning — regex-based removal of URLs, hashtags, mentions, special characters, and NLP-based stopword removal and lemmatization. |
| `resume_screening.ipynb` | Jupyter notebook containing the full ML pipeline: data loading, exploratory data analysis (EDA), text preprocessing, TF-IDF vectorization, model training, hyperparameter tuning, evaluation metrics, and model serialization. |
| `clf.pkl` | Pre-trained scikit-learn classifier model serialized with Python's `pickle` module for fast inference. |
| `tfidf.pkl` | Pre-fitted `TfidfVectorizer` object serialized with `pickle` to ensure consistent feature transformation at inference time. |
| `UpdatedResumeDataSet.csv` | The labeled dataset containing resumes and their corresponding job categories, used for training and evaluation. |
| `requirements.txt` | Lists all Python package dependencies required to run the project. |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit |
| **Machine Learning** | scikit-learn |
| **NLP** | NLTK (Natural Language Toolkit) |
| **Text Vectorization** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn (in notebook) |
| **Model Serialization** | Pickle |
| **PDF Parsing** | PyPDF2 / pdfminer (if applicable) |

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yogdeep2004/resume_screening_assistant.git
cd resume_screening_assistant
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Usage

### Using the Web App

1. **Launch** the Streamlit app with `streamlit run app.py`.
2. **Upload** a resume file (`.txt` or `.pdf`) using the file uploader widget.
3. **Wait** a moment for the model to process and classify the resume.
4. **View** the predicted job category displayed on the screen.

### Example

```
Upload:    john_doe_resume.pdf
Processing...
Predicted Category:  Data Science
```

### Using the Jupyter Notebook

To explore the model training pipeline, run the notebook:

```bash
jupyter notebook resume_screening.ipynb
```

This notebook walks through:
- **Data Loading and Exploration** — Understanding the dataset shape, distributions, and categories
- **Text Preprocessing** — Cleaning resumes using regex and NLP
- **Visualization** — Category distribution plots, word clouds, etc.
- **Feature Engineering** — TF-IDF vectorization
- **Model Training** — Training classifiers (e.g., KNN, SVM, Random Forest, etc.)
- **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix
- **Model Export** — Saving the best model and vectorizer as `.pkl` files

---

## Model Details

### Text Preprocessing (cleantext.py)

The `cleanResume()` function performs the following operations:

```python
def cleanResume(resumeText):
    # 1. Remove URLs
    # 2. Remove RT and cc mentions
    # 3. Remove hashtags and @mentions
    # 4. Remove special characters and punctuation
    # 5. Remove extra whitespace
    # 6. Convert to lowercase
    # 7. Remove stopwords
    # 8. Apply lemmatization
    return cleaned_text
```

### Feature Extraction

- **Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Purpose:** Converts cleaned resume text into a sparse numerical vector that captures the importance of each word relative to the entire corpus.
- **Configuration:** Pre-fitted on the training dataset and serialized as `tfidf.pkl`.

### Classification Model

- **Algorithm:** Trained and evaluated using scikit-learn classifiers. The best-performing model is serialized as `clf.pkl`.
- **Potential algorithms explored in notebook:**
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Multinomial Naive Bayes
  - Logistic Regression
- **Label Encoding:** Job categories are encoded as integers using `LabelEncoder`.

### Supported Job Categories (25+)

The model can classify resumes into categories such as:

| | | |
|---|---|---|
| Data Science | Java Developer | Web Designing |
| HR | Business Analyst | Network Security Engineer |
| Civil Engineer | Android Developer | Database Administrator |
| Python Developer | Testing | DevOps Engineer |
| UI/UX Designer | Mechanical Engineer | Electrical Engineering |
| Health and Fitness | Teacher | Arts |
| Sales | Automation Testing | SAP Developer |
| ETL Developer | Operations Manager | Hadoop |
| Chef | ... and more | |

---

## Dataset

- **File:** `UpdatedResumeDataSet.csv`
- **Source:** Publicly available resume dataset (commonly sourced from Kaggle)
- **Structure:**

| Column | Description |
|--------|-------------|
| `Category` | The job category label (target variable) |
| `Resume` | The raw resume text (feature) |

- **Size:** Approximately 960+ labeled resumes across 25+ job categories
- **Format:** CSV with two columns

---

## Future Enhancements

- [ ] **PDF Parsing Improvement** — Better extraction from complex PDF layouts and multi-column resumes
- [ ] **Deep Learning Models** — Integrate BERT/Transformers for improved text understanding and classification accuracy
- [ ] **Confidence Scores** — Display prediction probabilities for top-N categories
- [ ] **Keyword Extraction** — Highlight key skills and technologies found in the resume
- [ ] **Resume Scoring** — Score resumes against specific job descriptions
- [ ] **Batch Processing** — Upload and classify multiple resumes at once
- [ ] **Multi-Language Support** — Extend support beyond English resumes
- [ ] **Docker Deployment** — Containerize the application for easy deployment
- [ ] **Cloud Deployment** — Deploy on AWS, GCP, Azure, or Streamlit Cloud
- [ ] **Email Integration** — Auto-forward classified resumes to relevant HR teams
- [ ] **Authentication** — Add user login for enterprise use cases

---

## Contributing

Contributions are welcome. Here is how to get started:

1. **Fork** the repository.
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request.

### Contribution Ideas

- Improve the text preprocessing pipeline
- Add support for `.docx` file uploads
- Enhance the Streamlit UI/UX
- Add more evaluation metrics and visualizations
- Write unit tests

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use, modify, and distribute.
```

---

## Author

| | |
|---|---|
| **Name** | Yogdeep |
| **GitHub** | [@Yogdeep2004](https://github.com/Yogdeep2004) |

---

<div align="center">

**If you found this project useful, consider giving it a star on GitHub.**

Built with Python, Streamlit, and scikit-learn.

</div>

---
