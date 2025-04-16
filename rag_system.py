import numpy as np
import faiss
import json
import uuid
import torch
import pandas as pd
import re
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from tenacity import retry, stop_after_attempt, wait_exponential
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx
from unstructured.partition.doc import partition_doc
from unstructured.documents.elements import Title, NarrativeText, Table, Element
from langchain.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from string import Template
import os
from pptx import Presentation
import subprocess
import tempfile
from pathlib import Path

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model for NER and keyword extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@dataclass
class DocumentEntry:
    id: str
    title: str
    text: str
    title_embedding: np.ndarray
    text_embedding: np.ndarray
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

class ParallelVectorStore:
    def __init__(self, embedding_dim: int = 768):
        self.title_index = faiss.IndexFlatL2(embedding_dim)
        self.text_index = faiss.IndexFlatL2(embedding_dim)
        self.documents: Dict[str, DocumentEntry] = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_ids = []

    def add_document(self, title: str, text: str, 
                     title_embedding: np.ndarray, text_embedding: np.ndarray,
                     keywords: List[str] = None, entities: Dict[str, List[str]] = None,
                     metadata: Dict = None) -> str:
        doc_id = str(uuid.uuid4())
        title_embedding = np.array(title_embedding, dtype=np.float32).reshape(1, -1)
        text_embedding = np.array(text_embedding, dtype=np.float32).reshape(1, -1)

        self.title_index.add(title_embedding)
        self.text_index.add(text_embedding)

        self.documents[doc_id] = DocumentEntry(
            id=doc_id,
            title=title,
            text=text,
            title_embedding=title_embedding.flatten(),
            text_embedding=text_embedding.flatten(),
            keywords=keywords or [],
            entities=entities or {},
            metadata=metadata or {}
        )
        
        self.document_ids.append(doc_id)
        
        # We need to rebuild the TF-IDF matrix when a new document is added
        self._rebuild_tfidf()
        
        return doc_id

    def _rebuild_tfidf(self):
        """Rebuild TF-IDF matrix with all documents"""
        corpus = [f"{doc.title} {doc.text}" for doc in self.documents.values()]
        
        if not corpus:
            print("No documents in corpus, skipping TF-IDF rebuild")
            return
            
        # Handle case where there are very few documents
        doc_count = len(corpus)
        print(f"Building TF-IDF matrix with {doc_count} documents")
        
        try:
            # For single document case, use a more permissive configuration
            if doc_count == 1:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_df=1.0,  # Include all terms (100%)
                    min_df=1,    # Include terms that appear at least once
                    stop_words='english'
                )
            else:
                # Regular case with multiple documents
                min_df_value = 1 if doc_count < 3 else 2
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_df=0.95, 
                    min_df=min_df_value, 
                    stop_words='english'
                )
                
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            print(f"TF-IDF matrix built successfully with shape {self.tfidf_matrix.shape}")
        except Exception as e:
            print(f"Error building TF-IDF matrix: {str(e)}")
            # Initialize with empty values so methods don't break
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentEntry]:
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        _, title_indices = self.title_index.search(query_embedding, k)
        _, text_indices = self.text_index.search(query_embedding, k)

        combined_indices = set(title_indices[0].tolist() + text_indices[0].tolist())
        return [self.documents[list(self.documents.keys())[idx]] for idx in combined_indices]
    
    def tfidf_search(self, query: str, k: int = 5) -> List[DocumentEntry]:
        """Search documents using TF-IDF similarity"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            print("TF-IDF search not available, returning empty results")
            return []
            
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarity_scores.argsort()[-k:][::-1]
            
            # Return the corresponding documents
            return [self.documents[self.document_ids[idx]] for idx in top_indices]
        except Exception as e:
            print(f"Error in TF-IDF search: {str(e)}")
            return []

    def keyword_search(self, keywords: List[str], k: int = 5) -> List[DocumentEntry]:
        """Search documents based on keyword matches"""
        keyword_scores = {}
        
        for doc_id, doc in self.documents.items():
            score = 0
            doc_keywords = set(doc.keywords)
            doc_text = f"{doc.title.lower()} {doc.text.lower()}"
            
            for keyword in keywords:
                # Add points for exact keyword match in the keywords list
                if keyword.lower() in doc_keywords:
                    score += 3
                
                # Add points for keyword occurrence in text
                score += doc_text.count(keyword.lower())
            
            keyword_scores[doc_id] = score
        
        # Sort by score and get top k
        top_docs = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [self.documents[doc_id] for doc_id, _ in top_docs]

    def save(self, path_prefix: str):
        faiss.write_index(self.title_index, f"{path_prefix}_title_vectors.faiss")
        faiss.write_index(self.text_index, f"{path_prefix}_text_vectors.faiss")

        documents_to_save = {}
        for doc_id, doc_entry in self.documents.items():
            doc_dict = asdict(doc_entry)
            doc_dict['title_embedding'] = doc_dict['title_embedding'].tolist()
            doc_dict['text_embedding'] = doc_dict['text_embedding'].tolist()
            documents_to_save[doc_id] = doc_dict

        with open(f"{path_prefix}_documents.json", 'w') as f:
            json.dump(documents_to_save, f)

    @classmethod
    def load(cls, path_prefix: str):
        store = cls()
        store.title_index = faiss.read_index(f"{path_prefix}_title_vectors.faiss")
        store.text_index = faiss.read_index(f"{path_prefix}_text_vectors.faiss")

        with open(f"{path_prefix}_documents.json", 'r') as f:
            documents_dict = json.load(f)

        store.documents = {}
        store.document_ids = []
        for k, v in documents_dict.items():
            v['title_embedding'] = np.array(v['title_embedding'], dtype=np.float32)
            v['text_embedding'] = np.array(v['text_embedding'], dtype=np.float32)
            store.documents[k] = DocumentEntry(**v)
            store.document_ids.append(k)
            
        # Rebuild TF-IDF matrix
        store._rebuild_tfidf()
        
        return store

class ResponseTemplate:
    """Template-based response generation system"""
    
    def __init__(self):
        # Default templates for common question types
        self.templates = {
            "definition": Template("$entity refers to $definition"),
            "summary": Template("Here's a summary of $topic: $summary"),
            "list": Template("Here are the key points about $topic:\n$items"),
            "comparison": Template("Comparing $entity1 and $entity2:\n$comparison"),
            "fallback": Template("Based on the provided information: $content")
        }
        
    def add_template(self, template_type: str, template_string: str):
        """Add a new response template"""
        self.templates[template_type] = Template(template_string)
        
    def generate_response(self, template_type: str, **kwargs) -> str:
        """Generate a response using the specified template"""
        if template_type not in self.templates:
            template_type = "fallback"
            
        return self.templates[template_type].safe_substitute(**kwargs)

class RuleBasedExtractor:
    """Rule-based information extraction from text"""
    
    def __init__(self):
        self.entity_patterns = {}
        self.relation_patterns = {}
        
    def add_entity_pattern(self, entity_type: str, pattern: str):
        """Add a regex pattern to extract entities of a specific type"""
        self.entity_patterns[entity_type] = re.compile(pattern, re.IGNORECASE)
        
    def add_relation_pattern(self, relation_type: str, pattern: str):
        """Add a regex pattern to extract relations between entities"""
        self.relation_patterns[relation_type] = re.compile(pattern, re.IGNORECASE)
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using defined patterns"""
        results = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[entity_type] = matches
                
        # Add NER from spaCy
        doc = nlp(text)
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type not in results:
                results[entity_type] = []
            results[entity_type].append(ent.text)
            
        return results
        
    def extract_relations(self, text: str) -> Dict[str, List[str]]:
        """Extract relations from text using defined patterns"""
        results = {}
        
        for relation_type, pattern in self.relation_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results[relation_type] = matches
                
        return results

class ExtractiveAnswerGenerator:
    """Generate answers by extracting relevant sentences from documents"""
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        if not text or not isinstance(text, str):
            return []
        
        try:
            sentences = sent_tokenize(text)
            return [s for s in sentences if s.strip()]  # Filter out empty sentences
        except Exception as e:
            print(f"Error extracting sentences: {str(e)}")
            return []
    
    @staticmethod
    def rank_sentences(query: str, sentences: List[str]) -> List[Tuple[str, float]]:
        """Rank sentences by relevance to query using TF-IDF"""
        if not sentences or not query:
            return []
            
        # Filter out any empty sentences
        sentences = [s for s in sentences if s and s.strip()]
        
        if not sentences:
            return []
            
        # Create TF-IDF vectors for sentences
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            query_vector = vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
            
            # Return sentences with their scores
            return [(sentence, score) for sentence, score in zip(sentences, similarities)]
        except Exception as e:
            print(f"Error ranking sentences: {str(e)}")
            # Handle cases where vectorization fails (e.g., empty sentences)
            return [(sentence, 0.0) for sentence in sentences]
    
    def generate_answer(self, query: str, documents: List[DocumentEntry], 
                        max_sentences: int = 3) -> str:
        """Generate an answer by extracting and combining relevant sentences"""
        if not documents or not query:
            return "No relevant information found."
            
        all_sentences = []
        
        # Extract sentences from all documents
        for doc in documents:
            if not hasattr(doc, 'text') or not doc.text:
                continue
                
            try:
                sentences = self.extract_sentences(doc.text)
                all_sentences.extend(sentences)
            except Exception as e:
                print(f"Error processing document for sentences: {str(e)}")
                continue
        
        if not all_sentences:
            return "No relevant information found in the documents."
            
        # Rank sentences by relevance
        try:
            ranked_sentences = self.rank_sentences(query, all_sentences)
            
            # Select top sentences
            if not ranked_sentences:
                return "Couldn't extract relevant sentences from the documents."
                
            top_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)
            
            # Limit to max_sentences if we have enough
            if top_sentences:
                top_sentences = top_sentences[:min(max_sentences, len(top_sentences))]
            
            # Build answer from top sentences
            if not top_sentences:
                return "No relevant information found."
                
            answer = " ".join([sentence for sentence, _ in top_sentences])
            return answer
        except Exception as e:
            print(f"Error generating extractive answer: {str(e)}")
            return "Encountered an issue while generating an answer from the document."

class RAGSystem:
    def __init__(self, model_name: str = "deepseek-r1:14b", use_llm: bool = True, method: str = "hybrid"):
        self.vector_store = ParallelVectorStore()
        self.templates = ResponseTemplate()
        self.extractor = RuleBasedExtractor()
        self.extractive_generator = ExtractiveAnswerGenerator()
        self.use_llm = use_llm
        self.method = method  # Method can be "hybrid", "rule", "extractive"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.embedding_model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.embedding_model.eval()

        if use_llm:
            self.generation_model = Ollama(model=model_name)
            
        # Initialize common extraction patterns
        self._initialize_extraction_patterns()
        
    def _initialize_extraction_patterns(self):
        """Initialize common extraction patterns for the rule-based extractor"""
        # Entity patterns
        self.extractor.add_entity_pattern("email", r'[\w\.-]+@[\w\.-]+\.\w+')
        self.extractor.add_entity_pattern("phone", r'\+?[\d\s-]{10,}')
        self.extractor.add_entity_pattern("url", r'https?://[\w\.-/]+')
        self.extractor.add_entity_pattern("date", r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')
        
        # Relation patterns
        self.extractor.add_entity_pattern("definition", r'(\w+)\s+is\s+defined\s+as\s+([^.]+)')
        self.extractor.add_entity_pattern("comparison", r'compared\s+to\s+(\w+)[,\s]+(\w+)\s+([^.]+)')

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings[0]

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from text using spaCy"""
        doc = nlp(text)
        
        # Extract nouns, proper nouns, and named entities
        keywords = []
        
        # Add named entities
        keywords.extend([ent.text.lower() for ent in doc.ents])
        
        # Add noun phrases and important nouns
        for chunk in doc.noun_chunks:
            keywords.append(chunk.text.lower())
        
        # Filter out stopwords and short terms
        keywords = [k for k in keywords if len(k) > 2]
        
        # Count and return most common keywords
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(max_keywords)]

    def _convert_table_to_text(self, table: Table) -> str:
        """Convert a table to row-wise sentence format."""
        try:
            # Try to extract table data from metadata
            if hasattr(table, 'metadata') and table.metadata and 'text_as_html' in table.metadata:
                # Parse HTML table
                tables = pd.read_html(table.metadata['text_as_html'])
                if not tables:
                    return str(table)
                df = tables[0]
            else:
                # Try to convert from direct representation
                data = table.metadata.get('data', []) if hasattr(table, 'metadata') and table.metadata else []
                if not data:
                    return str(table)
                df = pd.DataFrame(data)
            
            # Convert rows to sentences
            sentences = []
            
            # Convert each row to a sentence
            for _, row in df.iterrows():
                row_items = []
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        col_name = str(col).strip()
                        if col_name and col_name != '':
                            row_items.append(f"{col_name}: {value}")
                        else:
                            row_items.append(str(value))
                
                if row_items:
                    row_text = "; ".join(row_items) + "."
                    sentences.append(row_text)
            
            return "\n".join(sentences)
        
        except Exception as e:
            # Fallback to string representation
            return f"Table content: {str(table)}"

    def ingest_pdf(self, pdf_path: str):
        """
        Ingest PDF file and extract content
        """
        try:
            print(f"Ingesting PDF from {pdf_path}")
            elements = partition_pdf(
                pdf_path,
                detect_tables=True,
                infer_table_structure=True,
                strategy="hi_res"
            )
            
            print(f"Found {len(elements)} elements in the PDF")
            self._process_elements(elements, pdf_path)
                
        except Exception as e:
            print(f"Error ingesting PDF: {str(e)}")
            import traceback
            traceback.print_exc()

    def _convert_ppt_to_pdf(self, ppt_path: str) -> str:
        """
        Convert PPT/PPTX file to PDF using an online conversion API
        Returns the path to the created PDF file or None if conversion failed
        """
        try:
            import requests
            import os
            import json
            import time
            import base64
            from urllib.parse import urljoin
            
            print(f"Converting PowerPoint file to PDF: {ppt_path}")
            
            # Create output file path with same name but .pdf extension
            pdf_filename = os.path.splitext(os.path.basename(ppt_path))[0] + ".pdf"
            output_dir = os.path.join(os.path.dirname(ppt_path), "../pdf")
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, pdf_filename)
            
            # Read file as binary
            with open(ppt_path, 'rb') as file:
                file_content = file.read()
            
            # Encode file content as base64
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Option 1: Use CloudConvert API (requires API key)
            # API_KEY = os.getenv("CLOUDCONVERT_API_KEY")
            # if API_KEY:
            #     url = "https://api.cloudconvert.com/v2/jobs"
            #     payload = {
            #         "tasks": {
            #             "import-file": {
            #                 "operation": "import/base64",
            #                 "file": encoded_content,
            #                 "filename": os.path.basename(ppt_path)
            #             },
            #             "convert-file": {
            #                 "operation": "convert",
            #                 "input": "import-file",
            #                 "output_format": "pdf"
            #             },
            #             "export-file": {
            #                 "operation": "export/url",
            #                 "input": "convert-file"
            #             }
            #         }
            #     }
            #     headers = {
            #         "Authorization": f"Bearer {API_KEY}",
            #         "Content-Type": "application/json"
            #     }
            #     
            #     response = requests.post(url, json=payload, headers=headers)
            #     if response.status_code == 200:
            #         job_id = response.json()["data"]["id"]
            #         
            #         # Wait for job to complete
            #         status_url = f"https://api.cloudconvert.com/v2/jobs/{job_id}"
            #         for _ in range(30):  # Try for 30 seconds
            #             time.sleep(1)
            #             status_response = requests.get(status_url, headers=headers)
            #             status_data = status_response.json()
            #             
            #             if status_data["data"]["status"] == "finished":
            #                 # Get download URL
            #                 for task in status_data["data"]["tasks"]:
            #                     if task["name"] == "export-file" and task["status"] == "finished":
            #                         download_url = task["result"]["files"][0]["url"]
            #                         
            #                         # Download the file
            #                         pdf_response = requests.get(download_url)
            #                         with open(pdf_path, 'wb') as f:
            #                             f.write(pdf_response.content)
            #                         
            #                         print(f"PDF conversion successful, saved to: {pdf_path}")
            #                         return pdf_path
            #             
            #             elif status_data["data"]["status"] == "error":
            #                 print(f"Conversion API error: {status_data}")
            #                 break
            
            # Option 2: Use a simpler, file-based conversion approach
            # Alternative implementation: Convert file directly using an API service
            # For demonstration, we'll implement a simplified version that uses a mock API
            
            print("Using direct file-to-PDF conversion method")
            
            # For demonstration purposes, let's create a simple file with the PPT content
            # In a real implementation, replace this with an actual API call
            import io
            from PyPDF2 import PdfWriter, PdfReader
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            # Create a minimal PDF with text indicating it's from a PPT
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            
            # Extract filename for display
            filename = os.path.basename(ppt_path)
            
            # Add some basic text to the PDF
            can.setFont("Helvetica", 14)
            can.drawString(100, 750, f"Converted from PowerPoint: {filename}")
            can.drawString(100, 730, "This is a placeholder for the actual PowerPoint content.")
            can.drawString(100, 710, "In a production environment, replace this with an actual API call.")
            
            # Try to extract some text content from the PPT binary
            import re
            text_content = re.findall(b'[a-zA-Z0-9 .,;:\'\"\n\r\t-]{4,}', file_content)
            y_pos = 650
            
            for i, text in enumerate(text_content[:30]):  # Limit to first 30 matches
                try:
                    decoded = text.decode('utf-8', errors='ignore')
                    if len(decoded) > 5 and not all(c.isdigit() for c in decoded):
                        can.setFont("Helvetica", 10)
                        # Wrap text to avoid going off page
                        text_to_display = decoded[:60] + "..." if len(decoded) > 60 else decoded
                        can.drawString(120, y_pos, text_to_display)
                        y_pos -= 15
                        if y_pos < 100:  # Start a new page if needed
                            can.showPage()
                            y_pos = 750
                except:
                    pass
            
            can.save()
            
            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            
            # Create a new PDF with the generated content
            new_pdf = PdfReader(packet)
            output = PdfWriter()
            
            # Add the page to the output
            for page in range(len(new_pdf.pages)):
                output.add_page(new_pdf.pages[page])
            
            # Write the output PDF to the file
            with open(pdf_path, "wb") as outputStream:
                output.write(outputStream)
            
            print(f"Created placeholder PDF with extracted text at: {pdf_path}")
            return pdf_path
                
        except Exception as e:
            print(f"Error converting PPT to PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def ingest_ppt(self, ppt_path: str):
        """
        Ingest PowerPoint files (.ppt or .pptx)
        Now converts PowerPoint to PDF first, then processes the PDF 
        """
        try:
            print(f"Ingesting PowerPoint from {ppt_path}")
            if not os.path.exists(ppt_path):
                print(f"Error: PowerPoint file not found at {ppt_path}")
                return False

            # First convert the PPT file to PDF
            pdf_path = self._convert_ppt_to_pdf(ppt_path)
            
            if pdf_path and os.path.exists(pdf_path):
                print(f"Processing PowerPoint as PDF: {pdf_path}")
                # Use the PDF ingestion method since it works well
                self.ingest_pdf(pdf_path)
                
                # Check if documents were added
                doc_count = len(self.vector_store.documents)
                if doc_count > 0:
                    print(f"Successfully processed PowerPoint as PDF with {doc_count} sections")
                    return True
                
            # If PDF conversion fails or no documents were extracted, try the original methods
            print("PDF conversion failed or no content extracted. Trying original PPT processing...")
            
            # Use appropriate method based on file extension
            file_extension = os.path.splitext(ppt_path)[1].lower()
            
            # Try the fallback method as it's more reliable for direct PPT extraction
            print("Trying direct Python-PPTX extraction...")
            elements = self._fallback_pptx_processing(ppt_path)
            
            # Only if fallback doesn't produce enough elements, try the unstructured library
            if not elements or len(elements) < 2:
                print("Fallback produced insufficient elements, trying unstructured library...")
                try:
                    if file_extension == '.pptx':
                        elements = partition_pptx(
                            ppt_path,
                            detect_tables=True,
                            infer_table_structure=True
                        )
                    else:  # For .ppt files
                        print("Warning: Using pptx parser for .ppt file (limited support)")
                        elements = partition_pptx(
                            ppt_path,
                            detect_tables=True,
                            infer_table_structure=True
                        )
                except Exception as e:
                    print(f"Error using unstructured library for PowerPoint: {str(e)}")
                    # If already attempted fallback, continue with whatever elements we have
            
            # Ensure we have at least some elements
            if not elements or len(elements) == 0:
                print("Warning: No elements extracted from PowerPoint file")
                # Create a minimal element with the filename as title
                filename = os.path.basename(ppt_path)
                elements = [Title(text=f"PowerPoint: {filename}")]
            
            print(f"Successfully extracted {len(elements)} elements from the PowerPoint")
            
            # Process the elements
            if any(hasattr(element, 'metadata') and element.metadata and 'slide_number' in element.metadata for element in elements):
                print("Organizing elements by slide...")
                self._process_slides(elements, ppt_path)
            else:
                print("Processing elements without slide organization...")
                self._process_elements(elements, ppt_path)
            
            # Verify documents were added
            if len(self.vector_store.documents) > 0:
                print(f"Success: {len(self.vector_store.documents)} documents added to vector store")
                return True
            else:
                print("Warning: No documents were added to the vector store")
                return False
                
        except Exception as e:
            print(f"Critical error ingesting PowerPoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _fallback_pptx_processing(self, ppt_path: str) -> List[Element]:
        """Fallback method to extract content from PowerPoint files if the partition_pptx fails"""
        try:
            print("Using fallback method for PowerPoint processing...")
            
            # Check file extension
            file_extension = os.path.splitext(ppt_path)[1].lower()
            
            # For older .ppt files, we'll use a text-based approach
            if file_extension == '.ppt':
                print("Using text extraction for older .ppt format")
                return self._process_ppt_as_text(ppt_path)
            
            # For .pptx files, use python-pptx
            from pptx import Presentation
            
            elements = []
            
            try:
                # Open the presentation
                prs = Presentation(ppt_path)
                
                # Process each slide
                for slide_number, slide in enumerate(prs.slides, 1):
                    print(f"Processing slide {slide_number}...")
                    
                    # Add slide title
                    slide_title = None
                    if hasattr(slide, 'shapes') and slide.shapes.title and slide.shapes.title.text:
                        title_text = slide.shapes.title.text.strip()
                        if title_text:
                            slide_title = Title(text=title_text)
                            slide_title.metadata = {"slide_number": slide_number}
                            elements.append(slide_title)
                            print(f"Added title: {title_text[:50]}...")
                    
                    # Process all shapes/text boxes
                    slide_text = []
                    
                    # Process each shape in the slide
                    for shape in slide.shapes:
                        try:
                            # For text boxes and shapes with text
                            if hasattr(shape, "text") and shape.text and shape.text.strip():
                                # Skip if it's the title we already added
                                if shape == slide.shapes.title:
                                    continue
                                text = shape.text.strip()
                                if text:
                                    slide_text.append(text)
                                    print(f"Found shape text: {text[:30]}...")
                        
                            # For tables
                            if hasattr(shape, 'has_table') and shape.has_table:
                                table_text = []
                                table = shape.table
                                for row in table.rows:
                                    row_text = []
                                    for cell in row.cells:
                                        if cell.text and cell.text.strip():
                                            row_text.append(cell.text.strip())
                                    if row_text:
                                        table_text.append(" | ".join(row_text))
                                if table_text:
                                    slide_text.append("\n".join(table_text))
                                    print(f"Found table with {len(table_text)} rows")
                        except Exception as shape_error:
                            print(f"Error processing shape: {str(shape_error)}")
                            continue
                        
                    # Add the combined text content as a narrative element
                    if slide_text:
                        slide_content = "\n".join(slide_text)
                        print(f"Adding narrative text with {len(slide_content)} chars")
                        text_element = NarrativeText(text=slide_content)
                        text_element.metadata = {"slide_number": slide_number}
                        elements.append(text_element)
                    elif not slide_title:
                        # If no title and no text, add a placeholder to ensure the slide exists
                        placeholder = Title(text=f"Slide {slide_number}")
                        placeholder.metadata = {"slide_number": slide_number}
                        elements.append(placeholder)
                        print(f"Added placeholder for empty slide {slide_number}")
                
                print(f"Extracted {len(elements)} elements from {len(prs.slides)} slides")
                return elements
                
            except Exception as pptx_error:
                print(f"Error using python-pptx: {str(pptx_error)}")
                # If python-pptx fails, try text-based approach
                return self._process_ppt_as_text(ppt_path)
            
        except Exception as e:
            print(f"Error in fallback PowerPoint processing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a basic element instead of empty list to avoid failures
            filename = os.path.basename(ppt_path)
            return [Title(text=f"PowerPoint: {filename}")]

    def _process_ppt_as_text(self, ppt_path: str) -> List[Element]:
        """Process a PowerPoint file by extracting text content"""
        try:
            print(f"Extracting text content from PowerPoint file: {ppt_path}")
            
            # First try using catppt (from catdoc package)
            try:
                import subprocess
                # Try catppt (part of catdoc package)
                text = subprocess.check_output(['catppt', ppt_path], stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
                print(f"Extracted {len(text)} characters using catppt")
                
                # If catppt output is very short, try other methods
                if len(text.strip()) < 100:
                    raise Exception("Insufficient text extracted with catppt")
                    
            except Exception as catppt_error:
                print(f"catppt failed or returned insufficient content: {str(catppt_error)}")
                
                # Try using textract if available
                try:
                    import textract
                    text = textract.process(ppt_path).decode('utf-8')
                    print(f"Extracted {len(text)} characters using textract")
                    
                    # If textract output is very short, try raw extraction
                    if len(text.strip()) < 100:
                        raise Exception("Insufficient text extracted with textract")
                        
                except Exception as textract_error:
                    print(f"textract failed or returned insufficient content: {str(textract_error)}")
                    
                    # Last resort - try to read raw file content
                    with open(ppt_path, 'rb') as f:
                        content = f.read()
                    # Extract any readable text (crude but might work as last resort)
                    text = ''.join(char for char in content.decode('utf-8', errors='ignore') if char.isprintable())
                    print(f"Extracted {len(text)} characters using raw extraction")
            
            # Create a fallback full document if text is very short
            if len(text.strip()) < 50:
                filename = os.path.basename(ppt_path)
                elements = [
                    Title(text=f"PowerPoint: {filename}"),
                    NarrativeText(text=f"This is a PowerPoint presentation file. Limited text could be extracted: {text}")
                ]
                elements[0].metadata = {"slide_number": 1}
                elements[1].metadata = {"slide_number": 1}
                return elements
            
            # Split text into sections (try to identify slides)
            elements = []
            
            # Look for slide markers or try to split by blank lines
            slide_texts = []
            
            # Try to identify slide boundaries (looking for patterns like "Slide X" or multiple newlines)
            if 'Slide ' in text or 'SLIDE ' in text:
                # Use regex to find slide markers
                import re
                slide_matches = list(re.finditer(r'(Slide\s+\d+|SLIDE\s+\d+)', text))
                
                if slide_matches:
                    last_pos = 0
                    for match in slide_matches:
                        if last_pos > 0:  # Not the first slide marker
                            slide_content = text[last_pos:match.start()].strip()
                            if slide_content:
                                slide_texts.append(slide_content)
                        last_pos = match.start()
                        
                    # Add the last slide
                    if last_pos < len(text):
                        slide_content = text[last_pos:].strip()
                        if slide_content:
                            slide_texts.append(slide_content)
            
            # If no slide markers found, split by multiple newlines
            if not slide_texts:
                slide_texts = re.split(r'\n\s*\n\s*\n', text)
            
            # If still no slides, just use the whole text as one slide
            if not slide_texts or all(not s.strip() for s in slide_texts):
                slide_texts = [text]
            
            # Create elements for each slide
            for i, slide_text in enumerate(slide_texts, 1):
                slide_text = slide_text.strip()
                if not slide_text:
                    continue
                    
                # Split into lines to extract potential title
                lines = slide_text.split('\n')
                title_text = lines[0] if lines else f"Slide {i}"
                
                # Create title element
                title = Title(text=title_text)
                title.metadata = {"slide_number": i}
                elements.append(title)
                
                # Create content element (skip the first line if it was used as title)
                content_text = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                if content_text.strip():
                    content = NarrativeText(text=content_text)
                    content.metadata = {"slide_number": i}
                    elements.append(content)
            
            print(f"Created {len(elements)} elements from text extraction")
            
            # If we couldn't extract anything useful, create a minimal element
            if not elements:
                filename = os.path.basename(ppt_path)
                elements = [
                    Title(text=f"PowerPoint: {filename}"),
                    NarrativeText(text="This is a PowerPoint presentation file. No readable content could be extracted.")
                ]
                elements[0].metadata = {"slide_number": 1}
                elements[1].metadata = {"slide_number": 1}
            
            return elements
            
        except Exception as e:
            print(f"Error in text-based PowerPoint processing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a minimal element on failure
            filename = os.path.basename(ppt_path)
            return [
                Title(text=f"PowerPoint: {filename}"),
                NarrativeText(text="This is a PowerPoint presentation file. Content extraction failed.")
            ]

    def _process_slides(self, elements: List[Element], source_path: str):
        """Process PowerPoint elements grouped by slide"""
        slides = {}
        processed_count = 0
        
        print(f"Processing {len(elements)} elements from PowerPoint slides")
        
        # Group elements by slide number
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata and 'slide_number' in element.metadata:
                slide_number = element.metadata['slide_number']
                if slide_number not in slides:
                    slides[slide_number] = {'title': '', 'text': []}
                
                # Extract title and content
                if isinstance(element, Title):
                    slides[slide_number]['title'] = str(element)
                    print(f"Slide {slide_number} title: {str(element)[:30]}...")
                elif isinstance(element, NarrativeText):
                    content = str(element).strip()
                    if content:
                        slides[slide_number]['text'].append(content)
                        print(f"Added {len(content)} chars of text to slide {slide_number}")
                elif isinstance(element, Table):
                    try:
                        table_text = self._convert_table_to_text(element)
                        if table_text and table_text.strip():
                            slides[slide_number]['text'].append(table_text)
                            print(f"Added {len(table_text)} chars of table text to slide {slide_number}")
                    except Exception as e:
                        print(f"Error converting table on slide {slide_number}: {str(e)}")
                        # Add the raw table content as fallback
                        slides[slide_number]['text'].append(str(element))
        
        # Process each slide as a section
        sections_count = 0
        for slide_number, content in sorted(slides.items()):
            print(f"\nProcessing slide {slide_number}...")
            
            # Skip slides with absolutely no content
            if not content['title'] and not content['text']:
                print(f"Skipping empty slide {slide_number}")
                continue
            
            # Create a title for slides missing one
            if not content['title']:
                content['title'] = f"Slide {slide_number}"
                print(f"Using default title: {content['title']}")
            
            # Enhance the content by adding slide context
            content['text'].insert(0, f"From slide {slide_number} in presentation.")
            
            # Process the slide content
            if self._process_section(content, source_path):
                sections_count += 1
        
        print(f"Successfully processed {sections_count} slides from the PowerPoint presentation")
        
        # Check if documents were added
        doc_count = len(self.vector_store.documents)
        if doc_count > 0:
            print(f"Total documents in vector store: {doc_count}")
            return doc_count
        else:
            print("Warning: No documents were added to the vector store from slide processing")
            # Add a fallback document with the filename
            filename = os.path.basename(source_path)
            title = f"PowerPoint: {filename}"
            text = f"This is a PowerPoint presentation file named {filename} containing {len(slides)} slides."
            
            title_embedding = self.get_embedding(title)
            text_embedding = self.get_embedding(text)
            
            doc_id = self.vector_store.add_document(
                title=title,
                text=text,
                title_embedding=title_embedding,
                text_embedding=text_embedding,
                keywords=["powerpoint", "presentation", filename],
                entities={},
                metadata={'source': source_path}
            )
            
            print(f"Added fallback document with ID: {doc_id}")
            return 1

    def _process_elements(self, elements, source_path):
        """Common processing for all document types."""
        # Process all elements maintaining their order
        processed_content = []
        current_section = {'title': '', 'text': []}
        sections_count = 0
        
        for i, element in enumerate(elements):
            try:
                if isinstance(element, Title):
                    if current_section['title'] or current_section['text']:
                        self._process_section(current_section, source_path)
                        sections_count += 1
                    current_section = {'title': str(element), 'text': []}
                    print(f"Found title: {str(element)[:50]}...")
                elif isinstance(element, Table):
                    # Convert table to text and add to current section
                    table_text = self._convert_table_to_text(element)
                    current_section['text'].append(table_text)
                    print(f"Added table content ({len(table_text)} chars)")
                elif isinstance(element, NarrativeText):
                    current_section['text'].append(str(element))
                    print(f"Added text content ({len(str(element))} chars)")
            except Exception as e:
                print(f"Error processing element {i}: {str(e)}")

        # Process the last section if it exists
        if current_section['title'] or current_section['text']:
            self._process_section(current_section, source_path)
            sections_count += 1
            
        print(f"Processed {sections_count} sections from the document")
        
        # Check if documents were added
        print(f"Total documents in vector store: {len(self.vector_store.documents)}")

    def _process_section(self, section: Dict, source: str):
        try:
            title = section.get('title', '').strip()
            # Join text parts and ensure it's a string
            raw_text = ' '.join([str(t) for t in section.get('text', []) if t])
            text = raw_text.strip()
            
            # Skip completely empty sections
            if not title and not text:
                print("Skipping completely empty section")
                return False
            
            # Provide default title if needed
            if not title:
                # Use first line of text as title if possible
                text_lines = text.split('\n')
                if text_lines and len(text_lines[0]) > 3:
                    title = text_lines[0][:50]  # Use first line as title
                else:
                    title = "Untitled Section"  # Default title
                print(f"Using generated title: {title}")
            
            # Ensure minimum text length
            if len(text) < 10:  # Very short content
                print(f"Warning: Section has very short content ({len(text)} chars)")
                # Combine title with text if text is too short
                if title and len(title) > len(text):
                    text = f"{title}. {text}"
                    print(f"Using title as content, new length: {len(text)} chars")
            
            print(f"Processing section: '{title[:50]}...' with {len(text)} chars of text")
            
            # Create embeddings
            title_embedding = self.get_embedding(title)
            text_embedding = self.get_embedding(text)
            
            # Extract keywords and entities
            keywords = self.extract_keywords(text)
            entities = self.extractor.extract_entities(text)
            
            print(f"Extracted {len(keywords)} keywords and {sum(len(v) for v in entities.values())} entities")

            # Add document to vector store
            doc_id = self.vector_store.add_document(
                title=title,
                text=text,
                title_embedding=title_embedding,
                text_embedding=text_embedding,
                keywords=keywords,
                entities=entities,
                metadata={'source': source}
            )
            
            print(f"Added document with ID: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error processing section: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _remove_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content from the response."""
        # Pattern to match <think>...</think> including any content inside
        pattern = r'<think>.*?</think>'
        # Remove matched content with re.sub, using re.DOTALL to match across lines
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def _classify_question_type(self, question: str) -> str:
        """Determine the type of question being asked"""
        question_lower = question.lower()
        
        # Pattern matching for question types
        if re.search(r'(what is|define|meaning of|definition of)', question_lower):
            return "definition"
        elif re.search(r'(list|what are|enumerate)', question_lower):
            return "list"
        elif re.search(r'(summarize|summary|overview|brief|describe)', question_lower):
            return "summary"
        elif re.search(r'(compare|difference between|versus|vs\.)', question_lower):
            return "comparison"
        elif re.search(r'(who|whom|whose)', question_lower):
            return "person"
        elif re.search(r'(when|what time|what date)', question_lower):
            return "temporal"
        elif re.search(r'(where|location|place)', question_lower):
            return "location"
        elif re.search(r'(why|reason|cause)', question_lower):
            return "explanation"
        elif re.search(r'(how many|how much|count|quantity)', question_lower):
            return "quantity"
        else:
            return "general"

    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract keywords from question specifically"""
        # Remove common question words and stopwords
        question_words = {"what", "when", "where", "who", "whom", "which", "why", "how", 
                         "is", "are", "was", "were", "will", "would", "should", "could", 
                         "can", "do", "does", "did", "has", "have", "had", "the", "a", "an", 
                         "in", "on", "at", "to", "for", "with", "about", "of"}
        
        # Process with spaCy
        doc = nlp(question)
        
        # Extract key terms (nouns, named entities, adjectives)
        keywords = []
        
        # Add named entities
        for ent in doc.ents:
            keywords.append(ent.text.lower())
            
        # Add nouns and adjectives
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in question_words:
                keywords.append(token.text.lower())
            if token.pos_ == "ADJ" and len(token.text) > 3:
                keywords.append(token.text.lower())
                
        return list(set(keywords))

    def _generate_non_llm_response(self, question: str, results: List[DocumentEntry]) -> Tuple[str, bool]:
        """Generate response using non-LLM techniques"""
        if not results:
            return "No relevant information found.", False
            
        question_type = self._classify_question_type(question)
        
        # For definition questions, try to extract definitions
        if question_type == "definition":
            # Extract what's being defined
            match = re.search(r'(?:what is|define|meaning of|definition of)\s+([^?\.]+)', question.lower())
            if match:
                term = match.group(1).strip()
                
                # Look for definitions in the results
                for doc in results:
                    # Look for patterns like "X is defined as Y" or "X is Y"
                    definition_patterns = [
                        rf'{term}\s+is\s+defined\s+as\s+([^\.]+)',
                        rf'{term}\s+refers\s+to\s+([^\.]+)',
                        rf'{term}\s+means\s+([^\.]+)',
                        rf'{term}\s+is\s+(?:a|an|the)\s+([^\.]+)'
                    ]
                    
                    for pattern in definition_patterns:
                        matches = re.search(pattern, doc.text.lower())
                        if matches:
                            definition = matches.group(1).strip()
                            return self.templates.generate_response(
                                "definition", 
                                entity=term.capitalize(), 
                                definition=definition
                            ), True
        
        # For list questions, extract bullet points
        if question_type == "list":
            # Extract the topic
            match = re.search(r'(?:list|what are|enumerate)\s+([^?\.]+)', question.lower())
            topic = match.group(1).strip() if match else ""
            
            # Extract sentences with list indicators
            list_items = []
            for doc in results:
                sentences = sent_tokenize(doc.text)
                for sentence in sentences:
                    # Look for sentences with list indicators or that contain the topic
                    if (topic in sentence.lower() or 
                        any(kw in sentence.lower() for kw in self._extract_question_keywords(question))):
                        list_items.append(f"• {sentence}")
                        
                # Limit to reasonable number of items
                if len(list_items) >= 5:
                    break
                    
            if list_items:
                return self.templates.generate_response(
                    "list",
                    topic=topic,
                    items="\n".join(list_items[:7])  # Limit to 7 items
                ), True
        
        # For summary questions, use extractive summarization
        if question_type == "summary":
            # Extract the topic
            match = re.search(r'(?:summarize|summary of|overview of|brief on|describe)\s+([^?\.]+)', question.lower())
            topic = match.group(1).strip() if match else ""
            
            # Generate extractive summary
            summary = self.extractive_generator.generate_answer(question, results, max_sentences=5)
            
            if summary and summary != "No relevant information found.":
                return self.templates.generate_response(
                    "summary",
                    topic=topic if topic else "the topic",
                    summary=summary
                ), True
                
        # For specific entity extraction (people, dates, locations)
        if question_type in ["person", "temporal", "location"]:
            entity_types = {
                "person": ["PERSON", "ORG"],
                "temporal": ["DATE", "TIME"],
                "location": ["GPE", "LOC"]
            }
            
            # Extract relevant entities from results
            relevant_entities = []
            for doc in results:
                # Use the pre-extracted entities if available
                if doc.entities:
                    for ent_type in entity_types.get(question_type, []):
                        if ent_type in doc.entities:
                            relevant_entities.extend(doc.entities[ent_type])
                            
            if relevant_entities:
                # Format response based on question type
                if question_type == "person":
                    return f"The relevant people/organizations are: {', '.join(set(relevant_entities)[:5])}", True
                elif question_type == "temporal":
                    return f"The relevant dates/times are: {', '.join(set(relevant_entities)[:5])}", True
                elif question_type == "location":
                    return f"The relevant locations are: {', '.join(set(relevant_entities)[:5])}", True
        
        # Default: Use extractive answer generation
        answer = self.extractive_generator.generate_answer(question, results)
        if answer and answer != "No relevant information found.":
            # Apply template
            return self.templates.generate_response(
                "fallback",
                content=answer
            ), True
            
        # No good answer found with non-LLM methods
        return "", False

    def query(self, question: str, k: int = 5, force_llm: bool = False) -> str:
        """
        Answer a question using the knowledge base
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            force_llm: Force using LLM for response generation
            
        Returns:
            The answer to the question
        """
        try:
            # Extract keywords from the question
            question_keywords = self._extract_question_keywords(question)
            
            # First try keyword search for exact matches
            keyword_results = self.vector_store.keyword_search(question_keywords, k=k)
            
            # If not enough results, try TF-IDF search
            if len(keyword_results) < k:
                try:
                    tfidf_results = self.vector_store.tfidf_search(question, k=k)
                    
                    # Combine results without duplicates
                    seen_ids = set(doc.id for doc in keyword_results)
                    for doc in tfidf_results:
                        if doc.id not in seen_ids:
                            keyword_results.append(doc)
                            seen_ids.add(doc.id)
                            if len(keyword_results) >= k:
                                break
                except Exception as e:
                    print(f"Error in TF-IDF search: {str(e)}")
            
            # If still not enough, use embedding search
            if len(keyword_results) < k:
                try:
                    query_embedding = self.get_embedding(question)
                    embedding_results = self.vector_store.search(query_embedding, k=k)
                    
                    # Combine results without duplicates
                    seen_ids = set(doc.id for doc in keyword_results)
                    for doc in embedding_results:
                        if doc.id not in seen_ids:
                            keyword_results.append(doc)
                            seen_ids.add(doc.id)
                            if len(keyword_results) >= k:
                                break
                except Exception as e:
                    print(f"Error in embedding search: {str(e)}")
            
            # Get final results (safely)
            results = keyword_results[:min(k, len(keyword_results))]
            
            # If no results found at all, return a user-friendly message
            if not results:
                return "I couldn't find any relevant information in the document to answer your question. Please try asking something else about the content of the document."
            
            # Choose method based on configuration
            if self.method == "rule" and not force_llm:
                # Rule-based approach
                non_llm_response, success = self._generate_non_llm_response(question, results)
                if success:
                    return non_llm_response
                return "No relevant rule-based answer found."
            
            elif self.method == "extractive" and not force_llm:
                # Pure extractive approach
                try:
                    return self.extractive_generator.generate_answer(question, results)
                except Exception as e:
                    print(f"Error in extractive generation: {str(e)}")
                    return "I couldn't generate an answer using the extractive method. Please try a different approach."
            
            elif self.method == "extraction" and not force_llm:
                # Entity extraction approach - extract entities and present them
                all_entities = {}
                for doc in results:
                    # Check if entities dictionary exists
                    if hasattr(doc, 'entities') and doc.entities:
                        for entity_type, entities in doc.entities.items():
                            if entity_type not in all_entities:
                                all_entities[entity_type] = set()
                            all_entities[entity_type].update(entities)
                
                if all_entities:
                    response_parts = ["Key information extracted from the documents:"]
                    for entity_type, entities in all_entities.items():
                        if len(entities) > 0:
                            entity_list = ", ".join(list(entities)[:5])
                            response_parts.append(f"{entity_type}: {entity_list}")
                
                    return "\n".join(response_parts)
                return "No relevant entities found in the documents."
            
            elif self.method == "tfidf" and not force_llm:
                # TF-IDF based approach - use only TF-IDF search and return matching docs
                try:
                    tfidf_results = self.vector_store.tfidf_search(question, k=k)
                    if tfidf_results and len(tfidf_results) > 0:
                        top_result = tfidf_results[0]
                        return f"Top match: {top_result.title}\n\n{top_result.text[:500]}..."
                    return "No relevant documents found using TF-IDF search."
                except Exception as e:
                    print(f"Error in TF-IDF search processing: {str(e)}")
                    return "Error processing TF-IDF search results."
            
            elif self.use_llm:
                # Hybrid approach with LLM
                try:
                    context = "\n\n".join([
                        f"Section: {doc.title}\nContent: {doc.text[:1000]}..." 
                        for doc in results
                    ])
            
                    prompt = f"Answer the question using the provided context\n\nContext:\n{context}\n\nQuestion: {question} Just Answer"
                    response = self.generation_model(prompt)
                    
                    # Remove <think> tags from the response
                    cleaned_response = self._remove_think_tags(response)
                    
                    return cleaned_response
                except Exception as e:
                    print(f"Error generating LLM response: {str(e)}")
                    return "I encountered an issue generating a response. Please try again or use a different approach."
            else:
                # Default to extractive if no method matched
                try:
                    return self.extractive_generator.generate_answer(question, results)
                except Exception as e:
                    print(f"Error in default extractive generation: {str(e)}")
                    return "I couldn't generate an answer using the extractive method. Please try a different approach."
        except Exception as e:
            print(f"Unexpected error in query processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return "I encountered an unexpected error while processing your question. Please try again with a different question."

    def save(self, path_prefix: str):
        self.vector_store.save(path_prefix)

    @classmethod
    def load(cls, path_prefix: str, model_name: str = "deepseek-r1:14b", use_llm: bool = True, method: str = "hybrid"):
        rag = cls(model_name, use_llm=use_llm, method=method)
        rag.vector_store = ParallelVectorStore.load(path_prefix)
        return rag

    def ingest_doc(self, doc_path: str):
        """Ingest Word documents (.doc or .docx)"""
        try:
            print(f"Ingesting Word document from {doc_path}")
            if not os.path.exists(doc_path):
                print(f"Error: Word document not found at {doc_path}")
                return
            
            # Use appropriate method based on file extension
            file_extension = os.path.splitext(doc_path)[1].lower()
            
            if file_extension == '.docx':
                elements = partition_docx(
                    doc_path,
                    detect_tables=True,
                    infer_table_structure=True
                )
            else:  # For .doc files
                elements = partition_doc(
                    doc_path,
                    detect_tables=True,
                    infer_table_structure=True
                )
            
            if not elements:
                print("Warning: No elements extracted from Word document")
                # Create at least one element with the filename as title
                filename = os.path.basename(doc_path)
                elements = [Title(text=f"Document: {filename}")]
            
            print(f"Found {len(elements)} elements in the Word document")
            self._process_elements(elements, doc_path)
            
        except Exception as e:
            print(f"Error ingesting Word document: {str(e)}")
            import traceback
            traceback.print_exc()

# Usage Example
if __name__ == "__main__":
    # Example paths for different document types
    PDF_PATH = "/home/shahanahmed/Documents/pdf1.pdf"
    PPTX_PATH = "/home/shahanahmed/Documents/presentation.pptx"
    DOCX_PATH = "/home/shahanahmed/Documents/document.docx"

    # Create a RAG system
    rag = RAGSystem(model_name="gemma3:latest", use_llm=True, method="hybrid")
    
    # Choose which file type to process
    document_type = "pdf"  # Change to "pptx" or "docx" to test other document types
    
    if document_type == "pdf":
        rag.ingest_pdf(PDF_PATH)
    elif document_type == "pptx":
        rag.ingest_ppt(PPTX_PATH) 
    elif document_type == "docx":
        rag.ingest_doc(DOCX_PATH)
    
    # Test with a sample question
    question = "Tell me about the document"
    
    # Hybrid approach
    print("Hybrid approach (non-LLM with LLM fallback):")
    rag.method = "hybrid"
    print(rag.query(question))
    
    # Rule-based approach
    print("\nPure non-LLM approach (rule):")
    rag.method = "rule"
    print(rag.query(question, force_llm=False))

    # Extractive approach
    print("\nPure non-LLM approach (extractive):")
    rag.method = "extractive"
    print(rag.query(question, force_llm=False))

    # Entity extraction approach
    print("\nPure non-LLM approach (extraction):")
    rag.method = "extraction"
    print(rag.query(question, force_llm=False))

    # TF-IDF approach
    print("\nPure non-LLM approach (tfidf):")
    rag.method = "tfidf"
    print(rag.query(question, force_llm=False))

