import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from IPadvisor import IPAdvisor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_execution.log'),
        logging.StreamHandler()
    ]
)

class QueryExecutor:
    def __init__(self):
        self.advisor = IPAdvisor()
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_queries(self, query_file: str) -> List[str]:
        """Load queries from a JSON file."""
        with open(query_file, 'r') as f:
            return json.load(f)
    
    def execute_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Execute queries and collect results."""
        results = []
        
        for i, query in enumerate(queries, 1):
            logging.info(f"Processing query {i}/{len(queries)}: {query}")
            
            try:
                answer, retrieved_docs = self.advisor.process_query(query)
                
                result = {
                    "query_id": i,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "answer": answer,
                    "retrieved_documents": [
                        {
                            "rank": j + 1,
                            "metadata": doc.metadata,
                            "excerpt": doc.page_content[:200] + "..."
                        }
                        for j, doc in enumerate(retrieved_docs[:5])
                    ]
                }
                
                results.append(result)
                
                # Save individual result
                result_file = self.results_dir / f"query_{i:03d}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logging.info(f"Saved result for query {i} to {result_file}")
                
            except Exception as e:
                logging.error(f"Error processing query {i}: {str(e)}")
                continue
        
        return results
    
    def save_all_results(self, results: List[Dict[str, Any]]):
        """Save all results to a single JSON file."""
        output_file = self.results_dir / "all_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved all results to {output_file}")

def main():
    executor = QueryExecutor()
    
    # Load queries
    queries = executor.load_queries("queries/test_queries.json")
    
    # Execute queries and collect results
    results = executor.execute_queries(queries)
    
    # Save all results
    executor.save_all_results(results)

if __name__ == "__main__":
    main() 