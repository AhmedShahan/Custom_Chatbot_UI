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
from unstructured.documents.elements import Title, NarrativeText, Table, Element
from langchain.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from string import Template

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
        return sent_tokenize(text)
    
    @staticmethod
    def rank_sentences(query: str, sentences: List[str]) -> List[Tuple[str, float]]:
        """Rank sentences by relevance to query using TF-IDF"""
        if not sentences:
            return []
            
        # Create TF-IDF vectors for sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(sentences)
            query_vector = vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
            
            # Return sentences with their scores
            return [(sentence, score) for sentence, score in zip(sentences, similarities)]
        except:
            # Handle cases where vectorization fails (e.g., empty sentences)
            return [(sentence, 0.0) for sentence in sentences]
    
    def generate_answer(self, query: str, documents: List[DocumentEntry], 
                        max_sentences: int = 3) -> str:
        """Generate an answer by extracting and combining relevant sentences"""
        all_sentences = []
        
        # Extract sentences from all documents
        for doc in documents:
            sentences = self.extract_sentences(doc.text)
            all_sentences.extend(sentences)
            
        # Rank sentences by relevance
        ranked_sentences = self.rank_sentences(query, all_sentences)
        
        # Select top sentences
        top_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Build answer from top sentences
        if not top_sentences:
            return "No relevant information found."
            
        answer = " ".join([sentence for sentence, _ in top_sentences])
        return answer

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
        try:
            print(f"Ingesting PDF from {pdf_path}")
            elements = partition_pdf(
                pdf_path,
                detect_tables=True,
                infer_table_structure=True,
                strategy="hi_res"
            )
            
            print(f"Found {len(elements)} elements in the PDF")

            # Process all elements maintaining their order
            processed_content = []
            current_section = {'title': '', 'text': []}
            sections_count = 0
            
            for i, element in enumerate(elements):
                try:
                    if isinstance(element, Title):
                        if current_section['title'] or current_section['text']:
                            self._process_section(current_section, pdf_path)
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
                self._process_section(current_section, pdf_path)
                sections_count += 1
                
            print(f"Processed {sections_count} sections from the PDF")
            
            # Check if documents were added
            print(f"Total documents in vector store: {len(self.vector_store.documents)}")
                
        except Exception as e:
            print(f"Error ingesting PDF: {str(e)}")
            import traceback
            traceback.print_exc()

    def _process_section(self, section: Dict, source: str):
        try:
            title = section['title']
            text = ' '.join(section['text'])
            
            # Skip empty sections
            if not text.strip():
                print("Skipping empty section")
                return
                
            print(f"Processing section: '{title[:50]}...' with {len(text)} chars of text")
            
            title_embedding = self.get_embedding(title) if title else self.get_embedding("No title")
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
            
        except Exception as e:
            print(f"Error processing section: {str(e)}")
            import traceback
            traceback.print_exc()

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
        # Extract keywords from the question
        question_keywords = self._extract_question_keywords(question)
        
        # First try keyword search for exact matches
        keyword_results = self.vector_store.keyword_search(question_keywords, k=k)
        
        # If not enough results, try TF-IDF search
        if len(keyword_results) < k:
            tfidf_results = self.vector_store.tfidf_search(question, k=k)
            
            # Combine results without duplicates
            seen_ids = set(doc.id for doc in keyword_results)
            for doc in tfidf_results:
                if doc.id not in seen_ids:
                    keyword_results.append(doc)
                    seen_ids.add(doc.id)
                    if len(keyword_results) >= k:
                        break
        
        # If still not enough, use embedding search
        if len(keyword_results) < k:
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
        
        # Get final results
        results = keyword_results[:k]
        
        # Choose method based on configuration
        if self.method == "rule" and not force_llm:
            # Rule-based approach
            non_llm_response, success = self._generate_non_llm_response(question, results)
            if success:
                return non_llm_response
            return "No relevant rule-based answer found."
            
        elif self.method == "extractive" and not force_llm:
            # Pure extractive approach
            return self.extractive_generator.generate_answer(question, results)
            
        elif self.method == "extraction" and not force_llm:
            # Entity extraction approach - extract entities and present them
            all_entities = {}
            for doc in results:
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
            tfidf_results = self.vector_store.tfidf_search(question, k=k)
            if tfidf_results:
                top_result = tfidf_results[0]
                return f"Top match: {top_result.title}\n\n{top_result.text[:500]}..."
            return "No relevant documents found using TF-IDF search."
            
        elif self.use_llm:
            # Hybrid approach with LLM
            context = "\n\n".join([
                f"Section: {doc.title}\nContent: {doc.text[:1000]}..." 
                for doc in results
            ])
    
            prompt = f"Answer the question using the provided context\n\nContext:\n{context}\n\nQuestion: {question} Just Answer"
            response = self.generation_model(prompt)
            
            # Remove <think> tags from the response
            cleaned_response = self._remove_think_tags(response)
            
            return cleaned_response
        else:
            # Default to extractive if no method matched
            return self.extractive_generator.generate_answer(question, results)

    def save(self, path_prefix: str):
        self.vector_store.save(path_prefix)

    @classmethod
    def load(cls, path_prefix: str, model_name: str = "deepseek-r1:14b", use_llm: bool = True, method: str = "hybrid"):
        rag = cls(model_name, use_llm=use_llm, method=method)
        rag.vector_store = ParallelVectorStore.load(path_prefix)
        return rag

# Usage Example
if __name__ == "__main__":
    PDF_PATH = "/home/shahanahmed/Documents/pdf1.pdf"  

    # Create a single RAG system and ingest the PDF
    rag = RAGSystem(model_name="gemma3:latest", use_llm=True, method="hybrid")
    rag.ingest_pdf(PDF_PATH)
    
    # Set up different queries using different methods
    question = "Tell me about India"
    
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

