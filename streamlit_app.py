import streamlit as st 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile

import ResumeClassificationPipeline

st.set_page_config(
    page_title="Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    pipeline = ResumeClassificationPipeline()
    return pipeline

def main():
    st.title("📄 Resume Screening Assistant")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Single Resume", "Bulk Processing", "Analytics"])
    
    if page == "Single Resume":
        single_resume_page()
    elif page == "Bulk Processing":
        bulk_processing_page()
    else:
        analytics_page()

def single_resume_page():
    st.header("Single Resume Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a resume file", 
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, Word document, or text file"
        )
        
        # Text input as alternative
        st.subheader("Or Paste Resume Text")
        resume_text = st.text_area("Paste resume content here", height=200)
    
    with col2:
        st.subheader("Classification Results")
        
        if uploaded_file is not None or resume_text:
            pipeline = load_pipeline()
            
            with st.spinner("Analyzing resume..."):
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    with open(f"temp_{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    result = pipeline.process_single_resume(f"temp_{uploaded_file.name}")
                else:
                    result = pipeline.process_single_resume(resume_text, is_file_path=False)
            
            if result['error']:
                st.error(f"Error: {result['error']}")
            else:
                # Display results
                st.success(f"**Category:** {result['predicted_category'].title()}")
                
                # Confidence indicator
                conf_color = "green" if result['confidence_level'] == "HIGH" else "orange" if result['confidence_level'] == "MEDIUM" else "red"
                st.markdown(f"**Confidence:** <span style='color:{conf_color}'>{result['confidence']:.2f} ({result['confidence_level']})</span>", unsafe_allow_html=True)
                
                st.info(f"**Word Count:** {result['word_count']}")
                
                # Feature scores
                st.subheader("Keyword Analysis")
                feature_scores = {k.replace('_score', '').replace('_', ' ').title(): v 
                                for k, v in result['features'].items() if k.endswith('_score')}
                
                fig = px.bar(x=list(feature_scores.keys()), y=list(feature_scores.values()),
                           title="Keyword Scores by Category")
                st.plotly_chart(fig, use_container_width=True)

def bulk_processing_page():
    st.header("Bulk Resume Processing")
    
    uploaded_files = st.file_uploader(
        "Upload multiple resume files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Select multiple resume files to process at once"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} files uploaded**")
        
        if st.button("Process All Resumes", type="primary"):
            pipeline = load_pipeline()
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                result = pipeline.process_single_resume(f"temp_{uploaded_file.name}")
                results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            # Display results
            results_df = pd.DataFrame(results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                successful = len(results_df[results_df['error'].isna()])
                st.metric("Successful", successful)
            with col3:
                avg_conf = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.2f}")
            
            # Results table
            st.subheader("Classification Results")
            display_cols = ['filename', 'predicted_category', 'confidence_level', 'word_count']
            st.dataframe(results_df[display_cols], use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="resume_classification_results.csv",
                mime="text/csv"
            )

def analytics_page():
    st.header("Classification Analytics")
    
    # Load sample data for demo
    st.info("This page would show analytics from your processed resumes. Upload some resumes in the Bulk Processing page first.")
    
    # Demo chart
    demo_data = {
        'Category': ['Data Science', 'Software Dev', 'HR', 'Marketing', 'Finance'],
        'Count': [15, 25, 8, 12, 10]
    }
    
    fig = px.pie(values=demo_data['Count'], names=demo_data['Category'], 
                title="Resume Distribution by Category (Demo)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
