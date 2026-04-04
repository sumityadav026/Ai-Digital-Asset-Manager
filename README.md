AI Digital Asset Manager (AI-DAM)

An intelligent document management system that enables searching files based on meaning rather than exact file names.


Overview

Traditional file systems rely on filename or keyword-based search, which often fails when users do not remember exact file names. This project addresses that limitation by implementing semantic search using machine learning techniques.

The system allows users to retrieve documents using natural language queries. For example, searching for “offer letter salary” can return relevant documents even if the file name does not contain those exact words.


Key Features
	•	Semantic search based on meaning instead of keywords
	•	Fast similarity search using FAISS
	•	Document chunking for improved accuracy
	•	Duplicate file detection using hashing
	•	Support for PDF and DOCX files
	•	Hybrid search combining semantic similarity and keyword matching
	•	Clean and ranked results with preview snippets


Technologies Used
	•	Frontend: Streamlit
	•	Backend: Python
	•	Embeddings: Sentence Transformers
	•	Vector Database: FAISS
	•	File Processing: PyPDF2, python-docx
	•	Storage: Pickle and local file system


System Workflow
	1.	User uploads a document
	2.	Text is extracted from the document
	3.	The extracted text is divided into smaller chunks
	4.	Each chunk is converted into a vector embedding
	5.	Embeddings are stored in a FAISS index along with metadata
	6.	User enters a search query
	7.	The query is converted into an embedding
	8.	FAISS retrieves the most similar embeddings
	9.	Results are ranked using hybrid search (semantic + keyword matching)
	10.	The best matching result per file is displayed with a preview

What Makes This Project Unique
	•	Implements semantic search instead of traditional keyword-based search
	•	Uses chunking to improve retrieval accuracy for structured documents
	•	Combines semantic similarity with keyword-based ranking (hybrid search)
	•	Includes duplicate detection using file hashing
	•	Designed as a real-world document retrieval system

Future Improvements
	•	Integration with language models for question-answering over documents
	•	Image-based search using multimodal models
	•	Deployment on cloud platforms
	•	Multi-user support and access control
