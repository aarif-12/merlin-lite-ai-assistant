"""
YouTube Video Summarizer using LangChain and HuggingFace models
Extracts transcript and generates summaries in different formats
"""

import re
import logging
from typing import Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeSummarizer:
    def __init__(self):
        self.summarizer = None
        
    @st.cache_resource
    def load_summarizer(_self):
        """Load the summarization model with caching"""
        try:
            logger.info("Loading summarization model: facebook/bart-large-cnn")
            _self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
            logger.info("Summarization model loaded successfully")
            return _self.summarizer
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise e
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript from YouTube video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([item['text'] for item in transcript_list])
            return transcript
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}")
            return None
    
    def chunk_text(self, text: str, max_chunk_length: int = 1000) -> List[str]:
        """Split text into chunks for processing"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize text using the loaded model"""
        if self.summarizer is None:
            self.summarizer = self.load_summarizer()
        
        try:
            # Handle long text by chunking
            if len(text.split()) > 1000:
                chunks = self.chunk_text(text, 800)
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.split()) > 10:  # Only summarize meaningful chunks
                        summary = self.summarizer(
                            chunk,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        summaries.append(summary[0]['summary_text'])
                
                # Combine chunk summaries
                combined_summary = " ".join(summaries)
                
                # Final summarization if combined text is still long
                if len(combined_summary.split()) > 200:
                    final_summary = self.summarizer(
                        combined_summary,
                        max_length=max_length * 2,
                        min_length=min_length,
                        do_sample=False
                    )
                    return final_summary[0]['summary_text']
                else:
                    return combined_summary
            else:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            raise e
    
    def format_summary(self, summary: str, format_type: str) -> str:
        """Format summary based on the requested type"""
        if format_type == "Bullet Points":
            # Split summary into sentences and format as bullets
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            return "\n".join([f"• {sentence}." for sentence in sentences[:8]])
        
        elif format_type == "Paragraph":
            return summary
        
        elif format_type == "Section-wise":
            # Try to create sections based on content
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if len(sentences) >= 4:
                mid_point = len(sentences) // 2
                section1 = ". ".join(sentences[:mid_point]) + "."
                section2 = ". ".join(sentences[mid_point:]) + "."
                return f"**Key Points:**\n{section1}\n\n**Additional Details:**\n{section2}"
            else:
                return summary
        
        return summary
    
    def process_youtube_video(self, url: str, summary_format: str) -> Dict[str, str]:
        """Main function to process YouTube video and return summary"""
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"error": "Invalid YouTube URL. Please provide a valid YouTube video link."}
            
            # Get transcript
            transcript = self.get_transcript(video_id)
            if not transcript:
                return {"error": "Could not extract transcript. The video may not have captions available."}
            
            # Clean transcript
            transcript = re.sub(r'\[.*?\]', '', transcript)  # Remove timestamps
            transcript = re.sub(r'\s+', ' ', transcript).strip()  # Clean whitespace
            
            if len(transcript.split()) < 10:
                return {"error": "Transcript too short to summarize."}
            
            # Generate summary
            summary = self.summarize_text(transcript)
            
            # Format summary
            formatted_summary = self.format_summary(summary, summary_format)
            
            return {
                "summary": formatted_summary,
                "video_id": video_id,
                "transcript_length": len(transcript.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing YouTube video: {str(e)}")
            return {"error": f"An error occurred while processing the video: {str(e)}"}

# Global instance
youtube_summarizer = YouTubeSummarizer()
