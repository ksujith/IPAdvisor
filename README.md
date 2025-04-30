# IPAdvisor: Intelligent Patent Analysis System

IPAdvisor is a Retrieval-Augmented Generation (RAG) based system for intelligent patent analysis. It provides capabilities for prior art searches, patent content extraction, summarization, and similarity identification.

## Features

- **Prior Art Search**: Identify relevant prior art for patent applications
- **Content Extraction**: Extract and summarize key information from patents
- **Similarity Analysis**: Find similar patents based on technical content
- **Configurable Pipeline**: Easily adjustable parameters for retrieval and generation
- **Comprehensive Logging**: Detailed logging for analysis and debugging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IPDOC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

1. Initialize the IPAdvisor system:
```python
from IPadvisor import IPAdvisor
advisor = IPAdvisor()
```

2. Process queries:
```python
results = advisor.process_query("Describe innovative solar panel technologies")
```

## Configuration

Key configuration parameters in `IPadvisor.py`:
- `max_docs_to_index`: Maximum number of documents to index
- `batch_size`: Batch size for processing
- `embedding_model`: Model for document embedding
- `llm_model`: Language model for generation
- `max_context_length`: Maximum context length for generation
- `top_k`: Number of documents to retrieve

## Directory Structure

- `/assets`: Model cache and data files
- `/evaluation`: Query results and evaluation metrics
- `/logs`: System logs

## Evaluation

The system supports evaluation of:
1. Prior art search accuracy
2. Information extraction quality
3. Similarity matching precision

Results are stored in JSON format in the evaluation directory.

## Logging

Comprehensive logging is implemented with different levels:
- INFO: General operation information
- DEBUG: Detailed debugging information
- ERROR: Error messages and stack traces

Logs are stored in the `/logs` directory.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 