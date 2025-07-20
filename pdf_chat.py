"""
PDF Chat functionality using LangChain RAG pipeline
Extracts PDF content, creates embeddings, and enables Q&A
"""

import os
import logging
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Tuple
from transformers import pipeline
from models.embedding_model import embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChatBot:
    def __init__(self):
        self.qa_pipeline = None
        self.vector_store = None
        self.documents = []
        self.embeddings = []
        
    @st.cache_resource
    def load_qa_model(_self):
        """Load the QA model with caching"""
        try:
            logger.info("Loading QA model: deepset/roberta-base-squad2")
            _self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2",
                device=-1  # Use CPU
            )
            logger.info("QA model loaded successfully")
            return _self.qa_pipeline
        except Exception as e:
            logger.error(f"Error loading QA model: {str(e)}")
            # Fallback to a lighter model
            try:
                logger.info("Falling back to distilbert-base-cased-distilled-squad")
                _self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=-1
                )
                return _self.qa_pipeline
            except Exception as e2:
                logger.error(f"Error loading fallback model: {str(e2)}")
                raise e2
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Save uploaded file temporarily
            with open("temp_pdf.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Extract text using PyMuPDF
            doc = fitz.open("temp_pdf.pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            
            # Clean up temporary file
            if os.path.exists("temp_pdf.pdf"):
                os.remove("temp_pdf.pdf")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
        
        return chunks
    
    def create_vector_store(self, documents: List[str]) -> faiss.IndexFlatL2:
        """Create FAISS vector store from documents"""
        try:
            # Generate embeddings
            logger.info("Generating embeddings for documents...")
            embeddings = embedding_model.get_embeddings(documents)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            logger.info(f"Created vector store with {len(documents)} documents")
            return index, embeddings_array
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise e
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents using vector similarity"""
        try:
            if self.vector_store is None or len(self.documents) == 0:
                return []
            
            # Generate query embedding
            query_embedding = embedding_model.get_single_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.vector_store.search(query_vector, k)
            
            # Return documents with similarity scores
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity = 1 / (1 + distance)
                    results.append((self.documents[idx], similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def answer_question(self, question: str, context_docs: List[str]) -> Dict[str, any]:
        """Answer question using the QA model and retrieved context"""
        if self.qa_pipeline is None:
            self.qa_pipeline = self.load_qa_model()
        
        try:
            # Combine context documents
            context = " ".join(context_docs)
            
            # Limit context length to avoid model limits
            max_context_length = 2000
            if len(context) > max_context_length:
                context = context[:max_context_length]
            
            # Get answer from QA model
            result = self.qa_pipeline(
                question=question,
                context=context
            )
            
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "context_used": len(context_docs)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "I'm sorry, I couldn't process your question. Please try rephrasing it.",
                "confidence": 0.0,
                "context_used": 0
            }
    
    def process_pdf(self, pdf_file) -> Dict[str, any]:
        """Process uploaded PDF and create vector store"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_file)
            
            if len(text.strip()) < 100:
                return {"error": "PDF contains insufficient text content."}
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if len(chunks) == 0:
                return {"error": "Could not create text chunks from PDF."}
            
            # Store documents
            self.documents = chunks
            
            # Create vector store
            self.vector_store, self.embeddings = self.create_vector_store(chunks)
            
            return {
                "success": True,
                "num_chunks": len(chunks),
                "total_words": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {"error": f"Error processing PDF: {str(e)}"}
    
    def chat(self, question: str) -> Dict[str, any]:
        """Main chat function - retrieve relevant docs and answer question"""
        try:
            if not self.documents or self.vector_store is None:
                return {
                    "answer": "Please upload a PDF document first.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Retrieve relevant documents
            similar_docs = self.similarity_search(question, k=3)
            
            if not similar_docs:
                return {
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Extract just the document texts for QA
            context_docs = [doc for doc, score in similar_docs]
            
            # Answer the question
            qa_result = self.answer_question(question, context_docs)
            
            # Add source information
            qa_result["sources"] = [
                {"text": doc[:200] + "..." if len(doc) > 200 else doc, "similarity": score}
                for doc, score in similar_docs
            ]
            
            return qa_result
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }
    
    def clear_memory(self):
        """Clear stored documents and vector store"""
        self.documents = []
        self.embeddings = []
        self.vector_store = None
        logger.info("Cleared PDF chat memory")

# Global instance
pdf_chatbot = PDFChatBot()
