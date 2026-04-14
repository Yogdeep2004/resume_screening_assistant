from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List, Optional
from utils.parser import parse_file
from utils.scoring import calculate_similarity, extract_keywords
from ml.pipeline import pipeline

router = APIRouter()

@router.post("/analyze-resumes")
async def analyze_resumes(
    files: List[UploadFile] = File(...),
    job_description: Optional[str] = Form("")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    
    for file in files:
        try:
            content_bytes = await file.read()
            text = parse_file(file.filename, content_bytes)
            
            if not text.strip():
                continue
                
            # ML Prediction
            category = pipeline.predict_category(text)
            
            # Match score & keywords
            score = calculate_similarity(text, job_description)
            keywords = extract_keywords(text)
            
            results.append({
                "filename": file.filename,
                "predicted_category": category,
                "match_score": score,
                "keywords": keywords
            })
            
        except Exception as e:
            # Provide error response for specific file but continue
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    # Sort results by match score descending
    results = sorted(results, key=lambda x: x.get("match_score", 0), reverse=True)
    
    return {
        "status": "success",
        "job_description_provided": bool(job_description),
        "results": results
    }
