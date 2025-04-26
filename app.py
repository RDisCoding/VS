import streamlit as st
import os
import tempfile
import time
from transformers import pipeline, AutoTokenizer
import re
from langdetect import detect
from unidecode import unidecode

# Set page configuration
st.set_page_config(
    page_title="AI Lecture Summarizer",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("AI Lecture Summarizer")
st.markdown("""
Upload a video or audio lecture to generate detailed, structured notes.
This tool uses AI to transcribe and summarize educational content.
""")

# Sidebar with options
st.sidebar.header("Settings")

# Model selection
asr_model = st.sidebar.selectbox(
    "Speech Recognition Model",
    ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"],
    index=3
)

summarization_model = st.sidebar.selectbox(
    "Summarization Model",
    ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-base"],
    index=0
)

# Advanced settings with expander
with st.sidebar.expander("Advanced Settings"):
    max_summary_length = st.slider("Max Summary Length", 100, 500, 150)
    min_summary_length = st.slider("Min Summary Length", 30, 100, 50)
    chunk_size = st.slider("Text Chunk Size", 512, 2048, 1024)
    show_timestamps = st.checkbox("Include Timestamps", value=False)

# Helper functions
@st.cache_resource
def load_models(asr_model_name, summarization_model_name):
    """Load and cache the ASR and summarization models"""
    # Check if local model exists, otherwise download from HF
    asr_local_path = f"models/{asr_model_name.split('/')[-1]}"
    if os.path.exists(asr_local_path):
        asr = pipeline('automatic-speech-recognition', model=asr_local_path)
    else:
        asr = pipeline('automatic-speech-recognition', model=asr_model_name)
    
    # Same for summarization model
    sum_local_path = f"models/{summarization_model_name.split('/')[-1]}"
    if os.path.exists(sum_local_path):
        tokenizer = AutoTokenizer.from_pretrained(sum_local_path)
        summarizer = pipeline('summarization', model=sum_local_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
        summarizer = pipeline('summarization', model=summarization_model_name)
    
    return asr, tokenizer, summarizer

def preprocess_text(text, include_timestamps=False):
    """Clean and preprocess the transcribed text"""
    # Remove timestamps if not needed
    if not include_timestamps:
        text = re.sub(r'\[\d+:\d+\.\d+\]', '', text)
    
    # Detect language with fallback
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    
    # Normalize Unicode and convert to lowercase for non-logographic languages
    if lang not in ['zh-cn', 'zh-tw', 'ja', 'ko']:
        text = unidecode(text).lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove filler words (English only)
    if lang == 'en':
        filler_words = r"\b(um|uh|like|you know|ah|hmm)\b"
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
    
    # Remove repetitive phrases
    text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', text)
    
    return text, lang

def chunk_text(text, tokenizer, max_tokens=512): # Reduced max_tokens to 512
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

def format_summary(summary_text):
    """Format the summary with headers and bullet points"""
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary_text)
    
    # Group sentences into paragraphs
    paragraphs = []
    current_para = []
    
    for sentence in sentences:
        current_para.append(sentence)
        if len(current_para) >= 3:
            paragraphs.append(' '.join(current_para))
            current_para = []
    
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # Identify potential headers (shorter sentences that might be topics)
    result = []
    for i, para in enumerate(paragraphs):
        if i == 0 or len(para.split()) < 8:
            result.append(f"## {para}")
        else:
            result.append(para)
            # Add some bullet points for key points
            if len(para.split()) > 15 and i % 2 == 0:
                key_points = para.split('. ')[:2]
                result.append("\n* " + "\n* ".join(key_points))
    
    return "\n\n".join(result)

# Main function to process the uploaded file
def process_lecture(file_path, asr_model, tokenizer, summarizer, show_timestamps, max_len, min_len, chunk_size):
    # Process audio
    with st.spinner("Transcribing audio..."):
        transcription = asr(file_path, return_timestamps=show_timestamps)
    
    # Preprocess text
    with st.spinner("Cleaning transcript..."):
        clean_text, lang = preprocess_text(transcription['text'], show_timestamps)
    
    # Summarize text
    with st.spinner("Generating summary..."):
        chunks = chunk_text(clean_text, tokenizer, chunk_size)
        summaries = []
        
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            progress_bar.progress((i + 1) / len(chunks))
        
        final_summary = ' '.join(summaries)
        formatted_summary = format_summary(final_summary)
    
    return clean_text, formatted_summary, lang

# Main interface
upload_tab, demo_tab = st.tabs(["Upload Lecture", "Demo"])

with upload_tab:
    uploaded_file = st.file_uploader("Upload a video or audio lecture", 
                                   type=["mp4", "mp3", "wav", "m4a", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Process button
        if st.button("Process Lecture"):
            try:
                # Load models
                with st.spinner("Loading models..."):
                    asr, tokenizer, summarizer = load_models(asr_model, summarization_model)
                
                # Process lecture
                transcript, summary, lang = process_lecture(
                    temp_file_path, 
                    asr, 
                    tokenizer, 
                    summarizer,
                    show_timestamps,
                    max_summary_length,
                    min_summary_length,
                    chunk_size
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Transcript")
                    st.text_area("Full Transcript", transcript, height=400)
                    st.download_button(
                        "Download Transcript",
                        transcript,
                        file_name="lecture_transcript.txt"
                    )
                
                with col2:
                    st.subheader("Lecture Notes")
                    st.markdown(summary)
                    st.download_button(
                        "Download Notes",
                        summary,
                        file_name="lecture_notes.md"
                    )
                
                # Additional info
                st.info(f"Detected language: {lang}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            finally:
                # Clean up
                os.unlink(temp_file_path)

with demo_tab:
    st.markdown("""
    ## Demo Instructions
    
    1. Select sample lecture from the dropdown below
    2. Click "Run Demo" to see the AI in action
    3. Review generated notes and transcript
    
    Note: Demo uses pre-processed examples for faster results.
    """)
    
    demo_option = st.selectbox(
        "Select a sample lecture",
        ["Introduction to Machine Learning", "History of Art", "Basic Economics"]
    )
    
    if st.button("Run Demo"):
        with st.spinner("Processing demo lecture..."):
            # Simulate processing time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
            
            # Show demo results based on selection
            if demo_option == "Introduction to Machine Learning":
                demo_transcript = "Today we're going to talk about the basics of machine learning. Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data. There are several types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."
                demo_summary = """## Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data.

* Supervised learning uses labeled data
* Unsupervised learning finds patterns in unlabeled data

## Types of Machine Learning

Supervised learning works with labeled data where the algorithm learns to predict outputs based on inputs. Unsupervised learning works with unlabeled data to find patterns and relationships.

* Reinforcement learning uses reward signals
* Deep learning utilizes neural networks

## Applications

Machine learning has applications across various fields including healthcare, finance, transportation, and entertainment. The technology continues to evolve rapidly with advancements in computational power."""
            
            elif demo_option == "History of Art":
                demo_transcript = "In today's lecture, we'll explore the Renaissance period in European art. The Renaissance began in Italy in the 14th century and spread throughout Europe during the 15th and 16th centuries. It was characterized by a revival of classical learning and values."
                demo_summary = """## Renaissance Art History

The Renaissance was a period of artistic and cultural transformation that began in Italy during the 14th century and spread throughout Europe.

* Originated in Florence, Italy
* Spanned approximately the 14th through 17th centuries

## Key Characteristics

Renaissance art marked a shift from medieval traditions to more naturalistic representation, featuring linear perspective and anatomical accuracy. Artists studied classical antiquity and developed techniques to create more lifelike images.

* Emphasis on proportion and harmony
* Revival of classical themes and forms

## Notable Artists

Several master artists defined the Renaissance period through their revolutionary works and techniques. Leonardo da Vinci, Michelangelo, and Raphael formed the trinity of great masters of the High Renaissance period."""
            
            else:  # Economics
                demo_transcript = "Economics is the social science that studies the production, distribution, and consumption of goods and services. It focuses on the behavior and interactions of economic agents and how economies work."
                demo_summary = """## Fundamentals of Economics

Economics studies how societies allocate scarce resources to satisfy unlimited wants and needs.

* Microeconomics examines individual markets and decision-making
* Macroeconomics looks at economy-wide phenomena

## Economic Systems

Different economic systems organize production and distribution in various ways. Market economies rely on supply and demand with minimal government intervention, while command economies feature significant central planning.

* Capitalism emphasizes private ownership
* Socialism focuses on collective or governmental ownership

## Economic Indicators

Economists use various metrics to assess economic health including GDP, unemployment rates, inflation, and consumer confidence indexes. These indicators help guide policy decisions and business strategies."""
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transcript")
                st.text_area("Full Transcript", demo_transcript, height=400)
                st.download_button(
                    "Download Transcript",
                    demo_transcript,
                    file_name="demo_transcript.txt"
                )
            
            with col2:
                st.subheader("Lecture Notes")
                st.markdown(demo_summary)
                st.download_button(
                    "Download Notes",
                    demo_summary,
                    file_name="demo_notes.md"
                )

# Footer
st.markdown("---")
st.markdown("Video Lecture Summarizer | AI-Powered Note Generation")