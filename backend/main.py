import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze

app = FastAPI(title="Resume Screening Application", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["analyze"])

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "ml", "models")

@app.on_event("startup")
def startup_event():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODEL_DIR, "svm_model.pkl")):
        print("⚠️  Pre-trained models not found. Generating sample models via train.py ...")
        import subprocess
        train_script = os.path.join(BACKEND_DIR, "ml", "train.py")
        result = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Error training models:", result.stderr)
        else:
            # Reload models into the pipeline singleton now that they exist
            from ml.pipeline import pipeline
            pipeline.load_models()

@app.get("/")
def root():
    return {"message": "Resume Screening API is running ✅"}
