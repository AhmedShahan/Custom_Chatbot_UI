# rag_system.py
import numpy as np
import faiss
import json
import uuid
import torch
import pandas as pd
import re
from transformers import RobertaModel, RobertaTokenizer
from typing import Dict, List
from dataclasses import dataclass, asdict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table, Element
from langchain.llms import Ollama

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

class RAGSystem:
    def __init__(self, model_name: str = "deepseek-r1"):
        self.vector_store = ParallelVectorStore()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.embedding_model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.embedding_model.eval()

        self.generation_model = Ollama(model=model_name)

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

    def _remove_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags and their content from the response."""
        # Pattern to match <think>...</think> including any content inside
        pattern = r'<think>.*?</think>'
        # Remove matched content with re.sub, using re.DOTALL to match across lines
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def query(self, question: str, k: int = 5) -> str:
        query_embedding = self.get_embedding(question)
        results = self.vector_store.search(query_embedding, k)

        context = "\n\n".join([
            f"Section: {doc.title}\nContent: {doc.text[:1000]}..." 
            for doc in results
        ])

        prompt = f"Answer the question using the provided context\n\nContext:\n{context}\n\nQuestion: {question} Just Answer"
        response = self.generation_model(prompt)
        
        # Remove <think> tags from the response
        cleaned_response = self._remove_think_tags(response)
        
        return cleaned_response

    def save(self, path_prefix: str):
        self.vector_store.save(path_prefix)

    @classmethod
    def load(cls, path_prefix: str, model_name: str = "deepseek-chat"):
        rag = cls(model_name)
        rag.vector_store = ParallelVectorStore.load(path_prefix)
        return rag