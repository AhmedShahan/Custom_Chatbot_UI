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
from langchain_ollama import OllamaLLM  # Updated import
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
        # Only rebuild if we have enough documents
        if len(self.documents) > 1:
            self._rebuild_tfidf()
        
        return doc_id

    def _rebuild_tfidf(self):
        """Rebuild TF-IDF matrix with all documents"""
        corpus = [f"{doc.title} {doc.text}" for doc in self.documents.values()]
        
        if not corpus:
            return
            
        # Adjust parameters to work with any number of documents
        # Set min_df=1 to work with any corpus size
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=1.0,  # Accept terms in up to 100% of documents 
            min_df=1,    # Accept terms in at least 1 document
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentEntry]:
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Ensure k is not larger than the number of documents
        k = min(k, len(self.documents))
        
        if k == 0:
            return []

        _, title_indices = self.title_index.search(query_embedding, k)
        _, text_indices = self.text_index.search(query_embedding, k)

        combined_indices = set(title_indices[0].tolist() + text_indices[0].tolist())
        return [self.documents[list(self.documents.keys())[idx]] for idx in combined_indices if idx < len(self.documents)]
    
    def tfidf_search(self, query: str, k: int = 5) -> List[DocumentEntry]:
        """Search documents using TF-IDF similarity"""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or len(self.documents) == 0:
            return []
            
        # Ensure k is not larger than the number of documents
        k = min(k, len(self.documents))
        
        if k == 0:
            return []
            
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarity_scores.argsort()[-k:][::-1]
            
            # Return the corresponding documents
            return [self.documents[self.document_ids[idx]] for idx in top_indices if idx < len(self.document_ids)]
        except Exception as e:
            print(f"TF-IDF search error: {e}")
            return []

    def keyword_search(self, keywords: List[str], k: int = 5) -> List[DocumentEntry]:
        """Search documents based on keyword matches"""
        if not keywords or len(self.documents) == 0:
            return []
            
        # Ensure k is not larger than the number of documents
        k = min(k, len(self.documents))
        
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
            
        # Rebuild TF-IDF matrix if enough documents
        if len(store.documents) > 1:
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
        doc = nlp(text[:10000])  # Limit text to avoid memory issues
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
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        try:
            sentence_vectors = vectorizer.fit_transform(sentences)
            query_vector = vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
            
            # Return sentences with their scores
            return [(sentence, score) for sentence, score in zip(sentences, similarities)]
        except Exception as e:
            print(f"Sentence ranking error: {e}")
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
    def __init__(self, model_name: str = "deepseek-r1", use_llm: bool = True):
        self.vector_store = ParallelVectorStore()
        self.templates = ResponseTemplate()
        self.extractor = RuleBasedExtractor()
        self.extractive_generator = ExtractiveAnswerGenerator()
        self.use_llm = use_llm
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.embedding_model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.embedding_model.eval()

        if use_llm:
            try:
                self.generation_model = OllamaLLM(model=model_name)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}")
                self.use_llm = False
            
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
        
        # Add more rule patterns specific to rule-based approach
        self.extractor.add_entity_pattern("cause_effect", r'(?:because|due to|as a result of)\s+([^,\.]+)')
        self.extractor.add_entity_pattern("process_step", r'(?:step|first|second|third|next|finally|lastly)\s+([^,\.]+)')
        self.extractor.add_entity_pattern("requirement", r'(?:requires|requirement|necessary|needed|must have)\s+([^,\.]+)')
        self.extractor.add_entity_pattern("benefit", r'(?:benefit|advantage|useful for|helps with)\s+([^,\.]+)')

    def get_embedding(self, text: str) -> np.ndarray:
        # Handle empty or very short texts
        if not text or len(text.strip()) < 3:
            text = "empty document"
            
        # Truncate very long texts to avoid memory issues
        if len(text) > 10000:
            text = text[:10000]
            
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
        # Handle empty text
        if not text or len(text.strip()) < 3:
            return []
            
        # Truncate very long texts
        if len(text) > 10000:
            text = text[:10000]
            
        try:
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
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

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
            elements = partition_pdf(
                pdf_path,
                detect_tables=True,
                infer_table_structure=True,
                strategy="hi_res"
            )
    
            # Process all elements maintaining their order
            processed_content = []
            current_section = {'title': '', 'text': []}
            
            for element in elements:
                if isinstance(element, Title):
                    if current_section['title'] or current_section['text']:
                        self._process_section(current_section, pdf_path)
                    current_section = {'title': str(element), 'text': []}
                elif isinstance(element, Table):
                    # Convert table to text and add to current section
                    table_text = self._convert_table_to_text(element)
                    current_section['text'].append(table_text)
                elif isinstance(element, NarrativeText):
                    current_section['text'].append(str(element))
    
            if current_section['title'] or current_section['text']:
                self._process_section(current_section, pdf_path)
                
        except Exception as e:
            print(f"Error ingesting PDF {pdf_path}: {e}")

    def _process_section(self, section: Dict, source: str):
        title = section['title']
        text = ' '.join(section['text'])
        
        # Skip empty sections
        if not text.strip():
            return
            
        title_embedding = self.get_embedding(title) if title else self.get_embedding("No title")
        text_embedding = self.get_embedding(text)
        
        # Extract keywords and entities
        keywords = self.extract_keywords(text)
        entities = self.extractor.extract_entities(text)

        self.vector_store.add_document(
            title=title,
            text=text,
            title_embedding=title_embedding,
            text_embedding=text_embedding,
            keywords=keywords,
            entities=entities,
            metadata={'source': source}
        )

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
        
        try:
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
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            # Fallback to simple word splitting
            words = question.lower().split()
            return [w for w in words if w not in question_words and len(w) > 3]

    def _generate_rule_based_response(self, question: str, results: List[DocumentEntry]) -> str:
        """Generate response using rule-based pattern matching and template filling"""
        if not results:
            return "No relevant information found."
            
        question_type = self._classify_question_type(question)
        question_keywords = self._extract_question_keywords(question)
        
        # 1. Definition questions
        if question_type == "definition":
            for term in question_keywords:
                for doc in results:
                    # Try different definition patterns
                    patterns = [
                        rf'{re.escape(term)}\s+is\s+defined\s+as\s+([^\.]+)',
                        rf'{re.escape(term)}\s+refers\s+to\s+([^\.]+)',
                        rf'{re.escape(term)}\s+means\s+([^\.]+)',
                        rf'{re.escape(term)}\s+is\s+(?:a|an|the)\s+([^\.]+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, doc.text.lower(), re.IGNORECASE)
                        if match:
                            definition = match.group(1).strip()
                            return self.templates.generate_response(
                                "definition",
                                entity=term.capitalize(),
                                definition=definition
                            )
        
        # 2. List questions
        if question_type == "list":
            topic_pattern = r'(?:list|what are|enumerate)\s+([^?\.]+)'
            topic_match = re.search(topic_pattern, question.lower())
            topic = topic_match.group(1).strip() if topic_match else ""
            
            list_items = []
            for doc in results:
                # Look for bulleted or numbered lists in the text
                list_pattern = r'(?:•|\*|\d+\.)\s+([^\n]+)'
                matches = re.findall(list_pattern, doc.text)
                if matches:
                    for item in matches[:7]:  # Limit to 7 items
                        list_items.append(f"• {item}")
                
                # If no explicit list found, try to extract sentences that contain topic keywords
                if not list_items:
                    sentences = sent_tokenize(doc.text)
                    for sentence in sentences:
                        if (topic in sentence.lower() or 
                            any(kw in sentence.lower() for kw in question_keywords)):
                            list_items.append(f"• {sentence}")
                        if len(list_items) >= 7:
                            break
            
            if list_items:
                return self.templates.generate_response(
                    "list",
                    topic=topic or "relevant points",
                    items="\n".join(list_items[:7])
                )
        
        # 3. Comparison questions
        if question_type == "comparison":
            comparison_pattern = r'(?:compare|difference between|versus|vs\.)\s+([^\s]+)\s+(?:and|with|to)\s+([^\s?\.]+)'
            match = re.search(comparison_pattern, question.lower())
            
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()
                comparison_points = []
                
                for doc in results:
                    # Look for direct comparison sentences
                    patterns = [
                        rf'(?:difference|differences|compared|comparing|versus|vs\.)[^\.]*{re.escape(entity1)}[^\.]*{re.escape(entity2)}([^\.]+)',
                        rf'{re.escape(entity1)}[^\.]*(?:unlike|while|whereas|but|however)[^\.]*{re.escape(entity2)}([^\.]+)',
                        rf'{re.escape(entity2)}[^\.]*(?:unlike|while|whereas|but|however)[^\.]*{re.escape(entity1)}([^\.]+)'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, doc.text.lower(), re.IGNORECASE)
                        for m in matches:
                            comparison_points.append(f"• {m.strip()}")
                    
                    # If not enough direct comparisons, find sentences mentioning both entities
                    if len(comparison_points) < 3:
                        sentences = sent_tokenize(doc.text)
                        for sentence in sentences:
                            sentence_lower = sentence.lower()
                            if entity1 in sentence_lower and entity2 in sentence_lower:
                                comparison_points.append(f"• {sentence.strip()}")
                
                if comparison_points:
                    comparison_text = "\n".join(set(comparison_points)[:5])  # Remove duplicates
                    return self.templates.generate_response(
                        "comparison",
                        entity1=entity1.capitalize(),
                        entity2=entity2.capitalize(),
                        comparison=comparison_text
                    )
        
        # 4. Entity questions (who, when, where)
        if question_type in ["person", "temporal", "location"]:
            entity_types = {
                "person": ["PERSON", "ORG"],
                "temporal": ["DATE", "TIME"],
                "location": ["GPE", "LOC"]
            }
            
            target_types = entity_types.get(question_type, [])
            found_entities = []
            context_sentences = []
            
            for doc in results:
                # Look for entities in the document metadata
                if doc.entities:
                    for ent_type in target_types:
                        if ent_type in doc.entities:
                            found_entities.extend(doc.entities[ent_type])
                
                # Also look for context sentences containing these entities
                if found_entities:
                    sentences = sent_tokenize(doc.text)
                    for sentence in sentences:
                        if any(entity.lower() in sentence.lower() for entity in found_entities):
                            if any(kw in sentence.lower() for kw in question_keywords):
                                context_sentences.append(sentence)
                                break
            
            if found_entities:
                unique_entities = list(set(found_entities))[:5]  # Limit to top 5 unique entities
                entities_str = ", ".join(unique_entities)
                
                if context_sentences:
                    context = " ".join(context_sentences[:2])  # Provide some context
                    return f"Based on the document, {entities_str}. {context}"
                else:
                    return f"Based on the document: {entities_str}."
        
        # 5. Explanation questions (why, how)
        if question_type in ["explanation"]:
            explanation_sentences = []
            
            for doc in results:
                sentences = sent_tokenize(doc.text)
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Look for causal language
                    if any(marker in sentence_lower for marker in 
                           ["because", "since", "as a result", "therefore", "thus", 
                            "hence", "due to", "causes", "caused by", "reason"]):
                        if any(kw in sentence_lower for kw in question_keywords):
                            explanation_sentences.append(sentence)
                
                # Look for extracted cause-effect relationships
                cause_effect_entities = self.extractor.extract_entities(doc.text).get("cause_effect", [])
                if cause_effect_entities:
                    for entity in cause_effect_entities[:3]:
                        explanation_sentences.append(f"This occurs because {entity}.")
            
            if explanation_sentences:
                explanation = " ".join(explanation_sentences[:3])
                return f"Explanation: {explanation}"
        
        # 6. Quantity questions (how many, how much)
        if question_type == "quantity":
            quantity_pattern = r'(?:how many|how much|count of|number of)\s+([^?\.]+)'
            match = re.search(quantity_pattern, question.lower())
            
            if match:
                target = match.group(1).strip()
                for doc in results:
                    # Look for sentences with numbers and the target
                    sentences = sent_tokenize(doc.text)
                    for sentence in sentences:
                        if target in sentence.lower() and re.search(r'\d+', sentence):
                            # Extract the number and its context
                            return f"Based on the document: {sentence}"
        
        # 7. Generic response based on most relevant sentences
        # Fall back to extracting the most relevant sentences
        relevant_sentences = []
        for doc in results:
            sentences = sent_tokenize(doc.text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Check if sentence contains question keywords
                if any(kw in sentence_lower for kw in question_keywords):
                    relevant_sentences.append(sentence)
            
        if relevant_sentences:
            # Return the top 3 relevant sentences
            response = " ".join(relevant_sentences[:3])
            return f"Based on the document: {response}"
        
        # Ultimate fallback
        return f"Based on the document, I couldn't find specific information about {', '.join(question_keywords)}."

    def _generate_extractive_response(self, question: str, results: List[DocumentEntry]) -> str:
        """Generate response by extracting and reranking sentences from documents"""
        if not results:
            return "No relevant information found."
        
        # Extract sentences from all documents
        all_sentences = []
        
        for doc in results:
            sentences = self.extractive_generator.extract_sentences(doc.text)
            for sentence in sentences:
                # Add the sentence with its document info
                all_sentences.append({
                    'text': sentence,
                    'doc_id': doc.id,
                    'title': doc.title
                })
        
        # Rank sentences by relevance to the query
        sentence_texts = [s['text'] for s in all_sentences]
        ranked_sentences = self.extractive_generator.rank_sentences(question, sentence_texts)
        
        # Add rank scores back to sentence objects
        for i, (_, score) in enumerate(ranked_sentences):
            all_sentences[i]['score'] = score
        
        # Sort by score in descending order
        sorted_sentences = sorted(all_sentences, key=lambda x: x['score'], reverse=True)
        
        # Select top sentences (adjust the number as needed)
        top_n = min(5, len(sorted_sentences))
        top_sentences = sorted_sentences[:top_n]
        
        # Determine the question type to format the response appropriately
        question_type = self._classify_question_type(question)
        
        # Format the response based on question type
        if question_type == "definition":
            # For definition questions, prefer sentences that have definition structure
            definition_sentences = []
            for sentence in top_sentences:
                text = sentence['text'].lower()
                # Look for definition patterns
                if re.search(r'is defined as|refers to|means|is a|is an|is the', text):
                    definition_sentences.append(sentence)
            
            if definition_sentences:
                return definition_sentences[0]['text']
            else:
                return top_sentences[0]['text']
                
        elif question_type == "list":
            # For list questions, format as bullet points
            list_items = [f"• {s['text']}" for s in top_sentences]
            return "\n".join(list_items)
            
        elif question_type == "comparison":
            # For comparison questions, try to find sentences that cover both compared entities
            question_keywords = self._extract_question_keywords(question)
            comparison_sentences = []
            
            # Check which sentences contain multiple keywords
            for sentence in top_sentences:
                text = sentence['text'].lower()
                keyword_count = sum(1 for kw in question_keywords if kw in text)
                if keyword_count >= 2:  # Contains at least 2 of the keywords
                    comparison_sentences.append(sentence)
            
            if comparison_sentences:
                return " ".join([s['text'] for s in comparison_sentences[:3]])
            
        # Default: concatenate top sentences
        return " ".join([s['text'] for s in top_sentences])

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
        # Handle empty knowledge base
        if not self.vector_store.documents:
            return "The knowledge base is empty. Please ingest some documents first."
            
        # Extract keywords from the question
        question_keywords = self._extract_question_keywords(question)
        
        # First try keyword search for exact matches
        keyword_results = self.vector_store.keyword_search(question_keywords, k=k) if question_keywords else []
        
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
        
        # If LLM is requested and available, use it
        if force_llm and self.use_llm:
            context = "\n\n".join([
                f"Section: {doc.title}\nContent: {doc.text[:1000]}..." 
                for doc in results
            ])
    
            prompt = f"Answer the question using the provided context\n\nContext:\n{context}\n\nQuestion: {question} Just Answer"
            response = self.generation_model(prompt)
            
            # Remove <think> tags from the response
            cleaned_response = self._remove_think_tags(response)
            
            return cleaned_response
        
        # Otherwise use the specified non-LLM approach
        if self.model_name == "rule":
            return self._generate_rule_based_response(question, results)
        elif self.model_name == "extractive" or not self.use_llm:
            return self._generate_extractive_response(question, results)
        else:
            # If LLM was requested but not available, use extractive as fallback
            return self._generate_extractive_response(question, results)

    def save(self, path_prefix: str):
        self.vector_store.save(path_prefix)

    @classmethod
    def load(cls, path_prefix: str, model_name: str = "deepseek-chat", use_llm: bool = True):
        rag = cls(model_name, use_llm=use_llm)
        rag.vector_store = ParallelVectorStore.load(path_prefix)
        return rag




# Usage Example
if __name__ == "__main__":
    PDF_PATH = "/home/shahanahmed/Documents/pdf1.pdf"  

    # Create RAG system with LLM backup
    rag1 = RAGSystem(model_name="deepseek-r1", use_llm=True)
    rag1.ingest_pdf(PDF_PATH)
    
    # Create pure extractive non-LLM RAG system
    rag2 = RAGSystem(model_name="extractive", use_llm=False)
    rag2.ingest_pdf(PDF_PATH)
    
    # Create pure rule-based non-LLM RAG system
    rag3 = RAGSystem(model_name="rule", use_llm=False)
    rag3.ingest_pdf(PDF_PATH)
    
    question = "Tell me about the document"
    
    # Try LLM-based response
    print("LLM-based approach:")
    print(rag1.query(question))
    
    # Pure extractive non-LLM approach
    print("\nPure extractive non-LLM approach:")
    print(rag2.query(question))
    
    # Pure rule-based non-LLM approach
    print("\nPure rule-based non-LLM approach:")
    print(rag3.query(question))