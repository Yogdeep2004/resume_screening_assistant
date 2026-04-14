import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Curated skill ontology for direct matching across common domains
SKILL_ONTOLOGY = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "golang",
    "rust", "swift", "kotlin", "php", "scala", "perl", "matlab", "r",

    # Web / Frontend
    "react", "angular", "vue", "html", "css", "sass", "tailwind", "bootstrap",
    "nextjs", "next.js", "redux", "webpack", "graphql", "rest", "api", "restful",

    # Backend / Frameworks
    "node", "nodejs", "fastapi", "django", "flask", "spring", "express",
    "rails", "laravel", "asp.net", "microservices",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
    "elasticsearch", "cassandra", "dynamodb", "firebase",

    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "linux", "bash", "devops", "git", "github", "gitlab", "ansible",
    "nginx", "apache",

    # Data Science / ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "tensorflow", "pytorch", "keras", "scikit-learn", "opencv", "pandas",
    "numpy", "matplotlib", "seaborn", "data science", "statistics", "modeling",
    "feature engineering", "regression", "classification", "clustering",
    "neural network", "transformer", "bert", "cnn", "rnn", "lstm",

    # Data Engineering / Analytics
    "hadoop", "spark", "kafka", "airflow", "databricks", "snowflake",
    "tableau", "power bi", "excel", "looker", "data pipeline", "etl",
    "data warehouse", "data analysis", "sql", "bigquery",

    # Design
    "figma", "photoshop", "illustrator", "ux", "ui", "user experience",
    "user interface", "wireframe", "prototype", "adobe xd", "sketch",
    "design system", "typography",

    # Project Management / Agile
    "agile", "scrum", "kanban", "jira", "confluence", "product management",
    "project management", "sprint", "roadmap", "stakeholder",

    # HR / Recruitment
    "recruitment", "talent acquisition", "hr", "human resources",
    "onboarding", "payroll", "performance management", "employee relations",
    "benefits administration", "learning development",

    # Finance / Accounting
    "accounting", "finance", "auditing", "taxation", "budgeting",
    "financial analysis", "forecasting", "gaap", "ifrs", "excel",
    "quickbooks", "sap", "erp",

    # Sales / Marketing
    "sales", "marketing", "seo", "sem", "crm", "salesforce", "hubspot",
    "social media", "content marketing", "email marketing", "analytics",
    "google analytics", "b2b", "b2c",

    # Cybersecurity
    "cybersecurity", "penetration testing", "vulnerability", "siem", "firewall",
    "encryption", "incident response", "compliance", "iso 27001", "soc",

    # Soft Skills / General
    "leadership", "communication", "teamwork", "problem solving",
    "critical thinking", "time management", "collaboration",
}


def normalize_text(text: str) -> str:
    """Lemmatize and clean text for better vocabulary overlap."""
    # Truncate to prevent spaCy slowness on huge resumes
    doc = nlp(text.lower()[:15000])
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 1
    ]
    return " ".join(tokens)


def extract_keywords(text: str) -> list:
    """Extract skills from resume using spaCy POS + skill ontology."""
    text_lower = text.lower()
    found = set()

    # 1. Match against skill ontology directly
    for skill in SKILL_ONTOLOGY:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found.add(skill)

    # 2. Also add spaCy nouns/proper nouns as supplementary keywords
    doc = nlp(text[:5000])
    for token in doc:
        if (
            token.pos_ in ["NOUN", "PROPN"]
            and not token.is_stop
            and len(token.text) > 2
        ):
            found.add(token.lemma_.lower())

    # Return up to 20 most relevant
    return sorted(list(found))[:20]


def calculate_similarity(resume_text: str, jd_text: str) -> float:
    """
    Hybrid similarity:
      60% — Skill/keyword overlap (ontology-based, reliable)
      40% — TF-IDF cosine similarity on lemmatized text (scaled)
    """
    if not jd_text or not jd_text.strip():
        return 0.0

    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()

    # --- Component 1: Skill ontology overlap ---
    jd_skills = set()
    resume_skills = set()
    for skill in SKILL_ONTOLOGY:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, jd_lower):
            jd_skills.add(skill)
        if re.search(pattern, resume_lower):
            resume_skills.add(skill)

    if jd_skills:
        overlap = jd_skills.intersection(resume_skills)
        keyword_score = len(overlap) / len(jd_skills)
    else:
        # No skills detected in JD — fall back to generic noun overlap
        jd_nouns = set(extract_keywords(jd_text))
        resume_nouns = set(extract_keywords(resume_text))
        if jd_nouns:
            keyword_score = len(jd_nouns & resume_nouns) / len(jd_nouns)
        else:
            keyword_score = 0.0

    # --- Component 2: TF-IDF cosine on lemmatized text ---
    try:
        resume_norm = normalize_text(resume_text)
        jd_norm = normalize_text(jd_text)
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        vectors = tfidf.fit_transform([resume_norm, jd_norm])
        raw_cosine = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        # Raw cosine for relevant pairs is typically 0.05–0.35
        # Scale to 0–1 range by multiplying by 3, capped at 1.0
        tfidf_scaled = min(raw_cosine * 3.5, 1.0)
    except Exception:
        tfidf_scaled = 0.0

    # --- Hybrid weighted score ---
    hybrid = 0.60 * keyword_score + 0.40 * tfidf_scaled
    return round(hybrid * 100, 2)
