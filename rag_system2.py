# rag_without_llm.py
import numpy as np
import faiss
import json
import uuid
import torch
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table, Element

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class DocumentEntry:
    id: str
    title: str
    text: str
    title_embedding: np.ndarray
    text_embedding: np.ndarray
    metadata: Dict = None

class ParallelVectorStore:
    def __init__(self, embedding_dim: int = 768):
        self.title_index = faiss.IndexFlatL2(embedding_dim)
        self.text_index = faiss.IndexFlatL2(embedding_dim)
        self.documents: Dict[str, DocumentEntry] = {}

    def add_document(self, title: str, text: str, 
                     title_embedding: np.ndarray, text_embedding: np.ndarray,
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
            metadata=metadata or {}
        )
        return doc_id

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentEntry]:
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        _, title_indices = self.title_index.search(query_embedding, k)
        _, text_indices = self.text_index.search(query_embedding, k)

        combined_indices = set(title_indices[0].tolist() + text_indices[0].tolist())
        return [self.documents[list(self.documents.keys())[idx]] for idx in combined_indices]

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
        for k, v in documents_dict.items():
            v['title_embedding'] = np.array(v['title_embedding'], dtype=np.float32)
            v['text_embedding'] = np.array(v['text_embedding'], dtype=np.float32)
            store.documents[k] = DocumentEntry(**v)

        return store

class RAGSystemWithoutLLM:
    def __init__(self):
        self.vector_store = ParallelVectorStore()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.embedding_model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.embedding_model.eval()

        # Initialize question type patterns
        self.question_patterns = {
            "definition": r"what (is|are|does) .+(\bmean\b|\bdefinition\b)?",
            "process": r"how (to|do|does|can) .+",
            "comparison": r"(what is|what are) the (differences?|similarities?) between .+",
            "list": r"(what|list|name) .+ (types|examples|kinds|ways) .+",
            "factual": r"(when|where|who|why|which) .+"
        }

        # Information extraction patterns
        self.extraction_patterns = {
            "date": r"\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4}",
            "person": r"[A-Z][a-z]+ [A-Z][a-z]+",
            "quantity": r"\d+ [a-zA-Z]+",
            "percentage": r"\d+(\.\d+)?\s*%",
            "money": r"\$\d+(\.\d+)?",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        }

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
                if current_section['title']:
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

    def _process_section(self, section: Dict, source: str):
        title = section['title']
        text = ' '.join(section['text'])
        title_embedding = self.get_embedding(title) if title else self.get_embedding("No title")
        text_embedding = self.get_embedding(text)

        self.vector_store.add_document(
            title=title,
            text=text,
            title_embedding=title_embedding,
            text_embedding=text_embedding,
            metadata={'source': source}
        )

    def _classify_question_type(self, question: str) -> str:
        """Determine the type of question based on patterns."""
        question_lower = question.lower()
        
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, question_lower):
                return q_type
                
        return "general"

    def _determine_extraction_type(self, question: str) -> Optional[str]:
        """Determine what type of information to extract based on the question."""
        question_lower = question.lower()
        
        if "date" in question_lower or "when" in question_lower:
            return "date"
        elif "who" in question_lower or "person" in question_lower:
            return "person"
        elif "how many" in question_lower or "amount" in question_lower:
            return "quantity"
        elif "percent" in question_lower or "percentage" in question_lower:
            return "percentage"
        elif "money" in question_lower or "cost" in question_lower or "price" in question_lower:
            return "money"
        elif "email" in question_lower or "contact" in question_lower:
            return "email"
        elif "website" in question_lower or "url" in question_lower:
            return "url"
            
        return None

    # METHOD 1: Direct Retrieval Only
    def query_retrieval_only(self, question: str, k: int = 5) -> str:
        """Simply return the most relevant passages from retrieved documents."""
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, k)
        
        if not results:
            return "No relevant information found."
        
        # Format the answer with the most relevant text passages
        answers = []
        for doc in results:
            # Truncate text to reasonable length and add ellipsis if needed
            text_preview = doc.text[:500]
            if len(doc.text) > 500:
                text_preview += "..."
                
            answers.append(f"From {doc.title}:\n{text_preview}")
        
        return "\n\n".join(answers)

    # METHOD 2: Template-Based Responses
    def query_template_based(self, question: str) -> str:
        """Create pre-defined responses based on question type and retrieved information."""
        # Classify question type
        question_type = self._classify_question_type(question)
        
        # Get relevant information
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 3)
        
        if not results:
            return "No information found to answer your question."
        
        # Get the most relevant document
        doc = results[0]
        
        # Use templates based on question type
        if question_type == "definition":
            extracted_text = doc.text[:250]
            return f"Definition: {extracted_text}"
            
        elif question_type == "process":
            # Try to find sentences that describe steps
            steps = []
            sentences = sent_tokenize(doc.text)
            for i, sent in enumerate(sentences):
                if any(word in sent.lower() for word in ["first", "then", "next", "finally", "step"]):
                    steps.append(f"{i+1}. {sent}")
            
            if steps:
                return "Process steps:\n" + "\n".join(steps[:5])
            else:
                return f"Process information: {doc.text[:400]}"
                
        elif question_type == "comparison":
            return f"Comparison: {doc.text[:500]}"
            
        elif question_type == "list":
            # Try to extract list items
            list_items = re.findall(r'(?:^|\n)(?:\d+\.\s|\*\s|-\s|\(\d+\)\s)([^\n]+)', doc.text)
            if list_items:
                return "List items:\n" + "\n".join([f"• {item.strip()}" for item in list_items[:7]])
            else:
                return f"List information: {doc.text[:400]}"
                
        elif question_type == "factual":
            return f"Fact: {doc.text[:300]}"
            
        else:  # general
            return f"Information: {doc.text[:400]}"

    # METHOD 3: Extractive Summarization
    def query_extractive(self, question: str) -> str:
        """Extract the most relevant sentences from retrieved documents."""
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 3)
        
        if not results:
            return "No relevant information found."
            
        all_sentences = []
        for doc in results:
            sentences = sent_tokenize(doc.text)
            
            for sentence in sentences:
                if len(sentence) > 10:  # Filter short sentences
                    sent_embedding = self.get_embedding(sentence)
                    similarity = cosine_similarity([query_embedding], [sent_embedding])[0][0]
                    all_sentences.append((sentence, similarity, doc.title))
        
        # Sort by similarity and take top sentences
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = all_sentences[:3]
        
        if not top_sentences:
            return "No relevant sentences found."
            
        # Format the answer
        answer_parts = []
        for sentence, score, title in top_sentences:
            answer_parts.append(f"{sentence} (from: {title}, relevance: {score:.2f})")
        
        return "\n\n".join(answer_parts)

    # METHOD 4: Rule-Based Text Generation
    def query_rule_based(self, question: str) -> str:
        """Implement rule-based systems for generating responses from retrieved information."""
        # Analyze question
        question_lower = question.lower()
        
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 3)
        
        if not results:
            return "No relevant information found."
            
        doc = results[0]
        
        # Apply different rules based on question words
        if "what is" in question_lower or "what are" in question_lower:
            term = question_lower.replace("what is", "").replace("what are", "").strip()
            return f"The {term} refers to {doc.text[:200]}"
            
        elif "how to" in question_lower:
            action = question_lower.replace("how to", "").strip()
            return f"To {action}, you should: {doc.text[:300]}"
            
        elif "why" in question_lower:
            return f"Reason: {doc.text[:250]}"
            
        elif "when" in question_lower:
            # Try to find dates in the text
            date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|[A-Z][a-z]+ \d{1,2}, \d{4}"
            dates = re.findall(date_pattern, doc.text)
            if dates:
                return f"Date information: {', '.join(dates[:3])}\nContext: {doc.text[:200]}"
            else:
                return f"Temporal information: {doc.text[:250]}"
                
        elif "where" in question_lower:
            return f"Location information: {doc.text[:250]}"
            
        elif "who" in question_lower:
            # Try to find person names
            person_pattern = r"[A-Z][a-z]+ [A-Z][a-z]+"
            persons = re.findall(person_pattern, doc.text)
            if persons:
                return f"Person information: {', '.join(persons[:3])}\nContext: {doc.text[:200]}"
            else:
                return f"Person information: {doc.text[:250]}"
        
        # Default response
        return f"Information: {doc.text[:300]}"

    # METHOD 5: Information Extraction
    def query_information_extraction(self, question: str) -> str:
        """Extract specific information from retrieved documents using pattern matching."""
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 5)
        
        if not results:
            return "No relevant information found."
        
        # Determine what information to extract based on the question
        extraction_type = self._determine_extraction_type(question)
        
        if not extraction_type:
            return f"Could not determine specific information to extract. Here's some relevant text: {results[0].text[:300]}"
        
        # Extract information from retrieved documents
        extracted_info = []
        for doc in results:
            if extraction_type in self.extraction_patterns:
                matches = re.findall(self.extraction_patterns[extraction_type], doc.text)
                for match in matches:
                    if isinstance(match, tuple):  # Some regex patterns might return tuples
                        match = ''.join(match)
                    extracted_info.append((match, doc.title))
        
        if extracted_info:
            # Remove duplicates
            seen = set()
            unique_info = []
            for info, title in extracted_info:
                if info not in seen:
                    seen.add(info)
                    unique_info.append((info, title))
                    
            # Format the answer
            formatted_info = [f"• {info} (from: {title})" for info, title in unique_info[:5]]
            return f"Found {extraction_type} information:\n" + "\n".join(formatted_info)
        else:
            return f"No {extraction_type} information found in the relevant documents."

    # METHOD 6: TF-IDF Based Response Generation
    def query_tfidf(self, question: str) -> str:
        """Generate responses by identifying important terms from both the question and retrieved documents."""
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 3)
        
        if not results:
            return "No relevant information found."
            
        # Combine retrieved texts
        corpus = [doc.text for doc in results]
        corpus.append(question)
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get important terms from question
        feature_names = vectorizer.get_feature_names_out()
        question_features = tfidf_matrix[-1].toarray()[0]
        
        # Get indices of non-zero elements and sort them by value
        nonzero_indices = question_features.nonzero()[0]
        important_indices = sorted(nonzero_indices, key=lambda i: question_features[i], reverse=True)[:5]
        important_terms = [feature_names[idx] for idx in important_indices]
        
        # Construct answer from sentences containing important terms
        relevant_sentences = []
        for doc in results:
            sentences = sent_tokenize(doc.text)
            for sentence in sentences:
                if any(term in sentence.lower() for term in important_terms):
                    relevant_sentences.append((sentence, doc.title))
        
        if not relevant_sentences:
            return f"No sentences found containing the key terms: {', '.join(important_terms)}."
            
        # Format the answer
        answer_parts = [f"Based on key terms: {', '.join(important_terms)}"]
        for i, (sentence, title) in enumerate(relevant_sentences[:3]):
            answer_parts.append(f"{i+1}. {sentence} (from: {title})")
            
        return "\n\n".join(answer_parts)

    # METHOD 7: Keyword-Based Answering
    def query_keyword_based(self, question: str) -> str:
        """Extract answers based on keyword matching between questions and documents."""
        # Extract keywords from question
        stop_words = set(stopwords.words('english'))
        question_words = [w.lower() for w in nltk.word_tokenize(question) if w.isalnum()]
        keywords = [w for w in question_words if w not in stop_words]
        
        if not keywords:
            return "Could not identify keywords in your question."
            
        # Find documents with matching keywords
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, 5)
        
        if not results:
            return "No relevant information found."
            
        # Score sentences based on keyword presence
        scored_sentences = []
        for doc in results:
            sentences = sent_tokenize(doc.text)
            for sentence in sentences:
                score = sum(1 for keyword in keywords if keyword in sentence.lower())
                if score > 0:
                    scored_sentences.append((sentence, score, doc.title))
        
        if not scored_sentences:
            return f"No sentences found containing the keywords: {', '.join(keywords)}."
            
        # Return top-scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Format the answer
        answer_parts = [f"Based on keywords: {', '.join(keywords)}"]
        for i, (sentence, score, title) in enumerate(scored_sentences[:3]):
            answer_parts.append(f"{i+1}. {sentence} (from: {title}, keyword matches: {score})")
            
        return "\n\n".join(answer_parts)

    # Main query method that allows switching between different response generation methods
    def query(self, question: str, method: str = "extractive", k: int = 5) -> str:
        """Generate a response to the question using the specified method."""
        method = method.lower()
        
        if method == "retrieval":
            return self.query_retrieval_only(question, k)
        elif method == "template":
            return self.query_template_based(question)
        elif method == "extractive":
            return self.query_extractive(question)
        elif method == "rule":
            return self.query_rule_based(question)
        elif method == "extraction":
            return self.query_information_extraction(question)
        elif method == "tfidf":
            return self.query_tfidf(question)
        elif method == "keyword":
            return self.query_keyword_based(question)
        else:
            # Default to extractive method
            return self.query_extractive(question)

    def save(self, path_prefix: str):
        self.vector_store.save(path_prefix)

    @classmethod
    def load(cls, path_prefix: str):
        rag = cls()
        rag.vector_store = ParallelVectorStore.load(path_prefix)
        return rag


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = RAGSystemWithoutLLM()
    
    # Ingest a PDF
    # rag_system.ingest_pdf("sample_document.pdf")
    
    # Or load an existing vector store
    # rag_system = RAGSystemWithoutLLM.load("my_vector_store")
    
    # Example questions
    questions = [
        "What is the main purpose of this document?",
        "How does the system process PDF files?",
        "Who created this system?",
        "When was this technology developed?"
    ]
    
    # Example of using different methods
    methods = ["retrieval", "template", "extractive", "rule", "extraction", "tfidf", "keyword"]
    
    for question in questions:
        print(f"Question: {question}")
        
        for method in methods:
            print(f"\n--- Method: {method} ---")
            response = rag_system.query(question, method=method)
            print(response)
            
        print("\n" + "="*50 + "\n")
    
    # Save the vector store
    # rag_system.save("my_vector_store")