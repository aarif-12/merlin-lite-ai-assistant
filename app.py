"""
Merlin-Lite: AI Assistant with YouTube Summarization and PDF Chat
Built with Streamlit, LangChain, and HuggingFace models
"""

import streamlit as st
import os
from datetime import datetime
from yt_summary import youtube_summarizer
from pdf_chat import pdf_chatbot

# Page configuration
st.set_page_config(
    page_title="Merlin-Lite AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Merlin-like styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .success-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Merlin-Lite AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by LangChain & HuggingFace • YouTube Summarization • PDF Chat</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        selected_feature = st.radio(
            "Choose a feature:",
            ["🎥 YouTube Summarizer", "📄 Chat with PDF"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Merlin-Lite** is an AI assistant that helps you:
        - Summarize YouTube videos
        - Chat with PDF documents
        - All using free HuggingFace models!
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Models Used")
        st.markdown("""
        - **Summarization**: facebook/bart-large-cnn
        - **Q&A**: deepset/roberta-base-squad2
        - **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
        """)
    
    # Main content area
    if selected_feature == "🎥 YouTube Summarizer":
        youtube_summarizer_page()
    else:
        pdf_chat_page()

def youtube_summarizer_page():
    st.markdown("## 🎥 YouTube Video Summarizer")
    st.markdown("Extract and summarize YouTube video transcripts in different formats.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        youtube_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
    
    with col2:
        summary_format = st.selectbox(
            "Summary Format",
            ["Bullet Points", "Paragraph", "Section-wise"],
            help="Choose how you want the summary formatted"
        )
    
    # Process button
    if st.button("🎯 Summarize Video", type="primary", use_container_width=True):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
            return
        
        with st.spinner("🔄 Processing video... This may take a few minutes for the first run."):
            try:
                # Process the video
                result = youtube_summarizer.process_youtube_video(youtube_url, summary_format)
                
                if "error" in result:
                    st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    # Success - display results
                    st.markdown('<div class="success-box">✅ <strong>Summary generated successfully!</strong></div>', unsafe_allow_html=True)
                    
                    # Video info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Video ID", result["video_id"])
                    with col2:
                        st.metric("Transcript Length", f"{result['transcript_length']} words")
                    
                    # Summary
                    st.markdown("### 📝 Summary")
                    st.markdown(f'<div class="bot-message">{result["summary"]}</div>', unsafe_allow_html=True)
                    
                    # Download option
                    summary_text = f"YouTube Video Summary\nURL: {youtube_url}\nFormat: {summary_format}\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{result['summary']}"
                    st.download_button(
                        "📥 Download Summary",
                        summary_text,
                        file_name=f"youtube_summary_{result['video_id']}.txt",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ <strong>Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
    
    # Example section
    with st.expander("💡 Example URLs to try"):
        st.markdown("""
        Here are some example YouTube URLs you can test:
        - Educational content
        - Tech talks
        - Tutorials
        
        **Note:** The video must have captions/subtitles available for summarization to work.
        """)

def pdf_chat_page():
    st.markdown("## 📄 Chat with PDF")
    st.markdown("Upload a PDF document and ask questions about its content using RAG (Retrieval-Augmented Generation).")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_info" not in st.session_state:
        st.session_state.pdf_info = {}
    
    # File upload section
    st.markdown("### 📤 Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to chat with its content"
    )
    
    if uploaded_file is not None:
        if not st.session_state.pdf_processed or st.session_state.get("current_pdf") != uploaded_file.name:
            with st.spinner("🔄 Processing PDF... This may take a few minutes for the first run."):
                try:
                    # Process the PDF
                    result = pdf_chatbot.process_pdf(uploaded_file)
                    
                    if "error" in result:
                        st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {result["error"]}</div>', unsafe_allow_html=True)
                        st.session_state.pdf_processed = False
                    else:
                        # Success
                        st.markdown('<div class="success-box">✅ <strong>PDF processed successfully!</strong></div>', unsafe_allow_html=True)
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf = uploaded_file.name
                        st.session_state.pdf_info = result
                        st.session_state.chat_history = []  # Clear chat history for new PDF
                        
                        # Display PDF info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Document", uploaded_file.name)
                        with col2:
                            st.metric("Text Chunks", result["num_chunks"])
                        with col3:
                            st.metric("Total Words", result["total_words"])
                            
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <strong>Processing Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
                    st.session_state.pdf_processed = False
        else:
            # PDF already processed
            st.markdown('<div class="info-box">ℹ️ <strong>PDF is ready for questions!</strong></div>', unsafe_allow_html=True)
            if st.session_state.pdf_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document", uploaded_file.name)
                with col2:
                    st.metric("Text Chunks", st.session_state.pdf_info["num_chunks"])
                with col3:
                    st.metric("Total Words", st.session_state.pdf_info["total_words"])
    
    # Chat interface
    if st.session_state.pdf_processed:
        st.markdown("### 💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Chat History")
            for i, (question, answer, confidence) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message"><strong>AI:</strong> {answer}</div>', unsafe_allow_html=True)
                if confidence > 0:
                    st.caption(f"Confidence: {confidence:.2f}")
                st.markdown("---")
        
        # Question input
        question = st.text_input(
            "Ask a question about the PDF:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_button = st.button("🤔 Ask Question", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and question:
            with st.spinner("🔍 Searching document and generating answer..."):
                try:
                    # Get answer from chatbot
                    result = pdf_chatbot.chat(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((
                        question,
                        result["answer"],
                        result["confidence"]
                    ))
                    
                    # Display the new answer
                    st.markdown("#### Latest Answer")
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {question}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bot-message"><strong>AI:</strong> {result["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Show confidence and sources
                    if result["confidence"] > 0:
                        st.caption(f"Confidence: {result['confidence']:.2f}")
                    
                    if result.get("sources"):
                        with st.expander("📚 Source Context"):
                            for i, source in enumerate(result["sources"]):
                                st.markdown(f"**Source {i+1}** (Similarity: {source['similarity']:.3f})")
                                st.text(source["text"])
                                st.markdown("---")
                    
                    # Clear the input
                    st.rerun()
                    
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ <strong>Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
        
        elif ask_button and not question:
            st.warning("Please enter a question first.")
    
    else:
        st.markdown('<div class="info-box">ℹ️ <strong>Please upload a PDF document to start chatting!</strong></div>', unsafe_allow_html=True)
    
    # Tips section
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        **For better Q&A results:**
        - Ask specific questions about the document content
        - Use clear, complete sentences
        - Try different phrasings if you don't get good results
        - The AI works best with factual questions about the document
        
        **Supported PDF types:**
        - Text-based PDFs (not scanned images)
        - Academic papers, reports, manuals
        - Books and articles with clear text
        """)

if __name__ == "__main__":
    main()
