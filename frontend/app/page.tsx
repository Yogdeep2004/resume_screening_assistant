"use client";

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { UploadCloud, FileText, X, CheckCircle, BarChart3, TrendingUp, AlertCircle, Briefcase } from 'lucide-react';

interface Result {
  filename: string;
  predicted_category?: string;
  match_score?: number;
  keywords?: string[];
  error?: string;
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    }
  });

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const analyzeResumes = async () => {
    if (files.length === 0) {
      setError("Please upload at least one resume.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('job_description', jobDescription);

    try {
      const response = await axios.post('http://localhost:8000/api/analyze-resumes', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResults(response.data.results);
    } catch (err: any) {
      setError(err.response?.data?.detail || "An error occurred while analyzing resumes. Make sure the backend makes the prediction models available.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1 className="gradient-text" style={{ fontSize: '3rem', marginBottom: '1rem' }}>
          AI Resume Screening
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem' }}>
          Intelligently parse, predict, and match resumes against your job descriptions.
        </p>
      </header>

      <div className="main-grid">
        {/* Left Panel: Inputs */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="glass-panel">
            <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <UploadCloud size={20} /> Upload Resumes
            </h3>
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <UploadCloud size={40} className="dropzone-icon" />
              <p>Drag & drop PDFs, DOCX, or TXT here</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                or click to select files
              </p>
            </div>

            {files.length > 0 && (
              <div className="file-list">
                {files.map((file, i) => (
                  <div key={i} className="file-item">
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      <FileText size={16} /> {file.name}
                    </span>
                    <button className="remove-file" onClick={() => removeFile(i)}>
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="glass-panel">
            <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Briefcase size={20} /> Job Description
            </h3>
            <textarea
              className="textarea-input"
              placeholder="Paste the job description here to calculate match scores..."
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
            />
          </div>

          <button 
            className="btn" 
            onClick={analyzeResumes} 
            disabled={loading || files.length === 0}
            style={{ width: '100%', padding: '1rem', fontSize: '1.1rem' }}
          >
            {loading ? <span className="loader"></span> : <><BarChart3 size={20} /> Analyze Resumes</>}
          </button>

          {error && (
            <div style={{ color: 'var(--error-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px' }}>
              <AlertCircle size={20} /> {error}
            </div>
          )}
        </div>

        {/* Right Panel: Results */}
        <div className="glass-panel" style={{ minHeight: '500px' }}>
          <h2 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <TrendingUp size={24} className="gradient-text" /> Analysis Results
          </h2>

          {results.length === 0 && !loading && !error && (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', flexDirection: 'column', gap: '1rem' }}>
              <BarChart3 size={48} style={{ opacity: 0.2 }} />
              <p>Upload resumes and click analyze to see results here.</p>
            </div>
          )}

          <div className="results-list">
            {results.map((result, i) => (
              <div key={i} className="glass-panel result-card" style={{ background: 'rgba(0,0,0,0.2)' }}>
                {result.error ? (
                  <div>
                    <h4 style={{ color: 'var(--error-color)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <AlertCircle size={18} /> Error processing {result.filename}
                    </h4>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>{result.error}</p>
                  </div>
                ) : (
                  <>
                    <div className="result-card-header">
                      <div>
                        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <CheckCircle size={20} color="var(--success-color)" /> {result.filename}
                        </h3>
                        {result.predicted_category && (
                          <div style={{ marginTop: '0.5rem', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', background: 'rgba(99, 102, 241, 0.2)', padding: '0.25rem 0.75rem', borderRadius: '4px', fontSize: '0.9rem', color: '#a5b4fc' }}>
                            Predicted Role: <strong>{result.predicted_category}</strong>
                          </div>
                        )}
                      </div>
                      
                      {jobDescription && result.match_score !== undefined && (
                        <div style={{ textAlign: 'right', minWidth: '100px' }}>
                          <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: result.match_score > 70 ? 'var(--success-color)' : result.match_score > 40 ? 'var(--warning-color)' : 'var(--error-color)' }}>
                            {result.match_score}%
                          </div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Match Score</div>
                        </div>
                      )}
                    </div>
                    
                    {jobDescription && result.match_score !== undefined && (
                      <div>
                        <div className="progress-container">
                          <div className="progress-bar" style={{ width: `${result.match_score}%`, background: result.match_score > 70 ? 'var(--success-color)' : result.match_score > 40 ? 'var(--warning-color)' : 'var(--error-color)' }}></div>
                        </div>
                      </div>
                    )}

                    {result.keywords && result.keywords.length > 0 && (
                      <div style={{ marginTop: '1rem' }}>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Extracted Skills & Keywords:</div>
                        <div>
                          {result.keywords.map((kw, j) => (
                            <span key={j} className="tag">{kw}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
