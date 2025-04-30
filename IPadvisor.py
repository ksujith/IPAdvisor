# === 0. System Configuration ===
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ipadvisor.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    'max_docs_to_index': 10000,
    'batch_size': 500,
    'embedding_model': "all-mpnet-base-v2",
    'llm_model': "microsoft/phi-1.5",
    'max_context_length': 500,
    'max_new_tokens': 256,
    'top_k': 5,
    'cache_dir': "./model_cache",
    'faiss_index_path': "./faiss_patent_index.faiss",
    'evaluation_output': "./evaluation_results"
}

# Create necessary directories
os.makedirs(CONFIG['cache_dir'], exist_ok=True)
os.makedirs(CONFIG['evaluation_output'], exist_ok=True)

# === 1. Import Necessary Libraries ===
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class IPAdvisor:
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.generator = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

    def load_dataset(self) -> bool:
        """Load and prepare the BigPatent dataset."""
        try:
            logging.info("Loading BigPatent dataset...")
            self.ds = load_dataset("big_patent", "d", trust_remote_code=True)
            logging.info("Dataset loaded successfully.")
            return True
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return False

    def prepare_documents(self) -> List[Document]:
        """Prepare documents for indexing."""
        documents = []
        logging.info(f"Preparing up to {CONFIG['max_docs_to_index']} documents...")
        
        for i, record in enumerate(self.ds["train"]):
            if i >= CONFIG['max_docs_to_index']:
                break
            
            # Enhanced document content structure
            content = {
                'description': record.get('description', ''),
                'abstract': record.get('abstract', ''),
                'claims': record.get('claims', []),
                'publication_date': record.get('publication_date', '')
            }
            
            # Convert to string format for embedding
            content_str = f"Description: {content['description']}\n"
            content_str += f"Abstract: {content['abstract']}\n"
            content_str += f"Claims: {' '.join(content['claims']) if isinstance(content['claims'], list) else content['claims']}"
            
            metadata = {
                "doc_id": i,
                "publication_number": record.get('publication_number', 'N/A'),
                "publication_date": content['publication_date']
            }
            
            documents.append(Document(page_content=content_str, metadata=metadata))
            
            if (i + 1) % 1000 == 0:
                logging.info(f"Prepared {i + 1} documents...")
        
        logging.info(f"Total documents prepared: {len(documents)}")
        return documents

    def setup_retrieval_module(self, documents: Optional[List[Document]] = None) -> bool:
        """Set up the retrieval module with embeddings and vector store."""
        try:
            logging.info("Setting up embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name=CONFIG['embedding_model'])
            
            if os.path.exists(CONFIG['faiss_index_path']):
                logging.info("Loading existing FAISS index...")
                self.vectorstore = FAISS.load_local(
                    CONFIG['faiss_index_path'],
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            elif documents:
                logging.info("Building new FAISS index...")
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                self.vectorstore.save_local(CONFIG['faiss_index_path'])
            else:
                logging.error("No documents provided for new index creation.")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error in retrieval module setup: {e}")
            return False

    def setup_generation_module(self) -> bool:
        """Set up the generation module with the specified model."""
        try:
            logging.info("Loading tokenizer and model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                CONFIG['llm_model'],
                trust_remote_code=True,
                cache_dir=CONFIG['cache_dir']
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG['llm_model'],
                trust_remote_code=True,
                cache_dir=CONFIG['cache_dir'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
                max_length=1024,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return True
        except Exception as e:
            logging.error(f"Error in generation module setup: {e}")
            return False

    def process_query(self, query: str) -> Tuple[str, List[Document]]:
        """Process a query through the RAG pipeline."""
        try:
            # Retrieval Phase
            logging.info(f"Processing query: {query}")
            retrieved_docs = self.vectorstore.similarity_search(query, k=CONFIG['top_k'])
            logging.info(f"Retrieved {len(retrieved_docs)} documents")

            # Generation Phase
            context = retrieved_docs[0].page_content
            if len(context) > CONFIG['max_context_length']:
                context = context[:CONFIG['max_context_length']] + "..."

            prompt = self.construct_prompt(query, context)
            
            output = self.generator(
                prompt,
                max_new_tokens=CONFIG['max_new_tokens'],
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )

            answer = self.extract_answer(output[0]['generated_text'])
            
            # Log the result
            self.log_query_result(query, answer, retrieved_docs)
            
            return answer, retrieved_docs

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}", []

    def construct_prompt(self, query: str, context: str) -> str:
        """Construct the prompt for the generation model."""
        return (
            f"### Intellectual Property Assistant\n\n"
            f"**Query:** {query}\n\n"
            f"**Relevant Patent Excerpt:**\n"
            f"{context}\n\n"
            f"**Instructions:** Based on the provided patent excerpt, provide a concise answer. "
            f"If the excerpt lacks sufficient information, please state that.\n\n"
            f"**Answer:**"
        )

    def extract_answer(self, generated_text: str) -> str:
        """Extract the answer from the generated text."""
        answer_start = generated_text.find("**Answer:**")
        if answer_start != -1:
            return generated_text[answer_start + len("**Answer:**"):].strip()
        return generated_text.strip()

    def log_query_result(self, query: str, answer: str, retrieved_docs: List[Document]):
        """Log the query result for evaluation."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'retrieved_docs': [
                {
                    'metadata': doc.metadata,
                    'excerpt': doc.page_content[:200] + "..."
                }
                for doc in retrieved_docs
            ]
        }
        
        filename = os.path.join(
            CONFIG['evaluation_output'],
            f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"Query result logged to {filename}")

# Main execution
if __name__ == "__main__":
    advisor = IPAdvisor()
    
    # Initialize the system
    if not advisor.load_dataset():
        exit(1)
    
    # Prepare documents if needed
    if not os.path.exists(CONFIG['faiss_index_path']):
        documents = advisor.prepare_documents()
    else:
        documents = None
    
    # Set up modules
    if not advisor.setup_retrieval_module(documents):
        exit(1)
    if not advisor.setup_generation_module():
        exit(1)
    
    # Run sample queries
    sample_queries = [
        "Find patents related to innovative solar panel technologies that discuss photovoltaic materials.",
        "Search for innovations in battery management systems for electric vehicles.",
        "Are there any patents on using AI for drug discovery?",
        "Summarize the main points about the described invention.",
        "What is the purpose of the invention described in the most relevant document?",
        "How is the retrieved patent similar to a new application for wireless charging?"
    ]
    
    for query in sample_queries:
        answer, docs = advisor.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        print("\nRetrieved Documents:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"Doc {i}: {doc.metadata}")
