"""
Data Ingestion and Processing Layer (CocoIndex functionality)
Handles data ingestion, transformation, and incremental indexing
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

try:
    from cocoindex import Flow, LocalFile, SplitRecursively, SentenceTransformerEmbed, ExtractByLlm
    from cocoindex.targets import Postgres, Neo4j, Qdrant
    from cocoindex.llm import LlmSpec, LlmApiType
    COCOINDEX_AVAILABLE = True
except ImportError:
    COCOINDEX_AVAILABLE = False
    logging.warning("CocoIndex not available. Using fallback implementation.")

from .config import UnifiedConfig

@dataclass
class ProcessedData:
    """Container for processed data"""
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    structured_data: Optional[Dict[str, Any]] = None

class DataIngestionEngine:
    """Handles data ingestion and processing using CocoIndex principles"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_components()
    
    def _setup_components(self):
        """Initialize processing components"""
        if COCOINDEX_AVAILABLE:
            self.flow = Flow()
            self.embedder = SentenceTransformerEmbed(model=self.config.repohyper.embedding_model)
            
            # Setup LLM spec for extraction
            self.llm_spec = LlmSpec(
                api_type=LlmApiType[self.config.llm_api_type],
                model=self.config.llm_model
            )
        else:
            self.logger.warning("Using fallback implementations")
            self.embedder = None
            self.llm_spec = None
    
    def ingest_code_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Ingest and process code repository
        Returns processed code data with embeddings and metadata
        """
        self.logger.info(f"Processing code repository: {repo_path}")
        
        if COCOINDEX_AVAILABLE:
            return self._process_code_with_cocoindex(repo_path)
        else:
            return self._process_code_fallback(repo_path)
    
    def _process_code_with_cocoindex(self, repo_path: str) -> Dict[str, Any]:
        """Process code using CocoIndex"""
        # Define data source
        code_source = LocalFile(path=repo_path, pattern="**/*.{py,js,ts,java,cpp,c}")
        
        # Create processing pipeline
        code_flow = self.flow.import_from(code_source)
        
        # Split code into meaningful chunks
        code_chunks = code_flow.transform(
            SplitRecursively(
                language="auto",
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )
        
        # Generate embeddings
        embedded_chunks = code_chunks.transform(
            self.embedder
        )
        
        # Export to vector store (Qdrant)
        embedded_chunks.export_to(
            Qdrant(
                url=self.config.cocoindex.qdrant_url,
                collection_name="code_chunks"
            )
        )
        
        # Export to graph database (Neo4j)
        embedded_chunks.export_to(
            Neo4j(
                url=self.config.cocoindex.neo4j_url,
                user=self.config.cocoindex.neo4j_user,
                password=self.config.cocoindex.neo4j_password
            )
        )
        
        # Execute the flow
        self.flow.run()
        
        return {"status": "success", "processed_files": len(code_flow)}
    
    def _process_code_fallback(self, repo_path: str) -> Dict[str, Any]:
        """Fallback implementation when CocoIndex is not available"""
        # Simple file processing without advanced chunking
        processed_files = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Simple chunking by lines
                        chunks = [content[i:i+self.config.chunk_size] 
                                 for i in range(0, len(content), self.config.chunk_size)]
                        
                        processed_files.append({
                            "file_path": file_path,
                            "chunks": chunks,
                            "metadata": {
                                "file_type": file.split('.')[-1],
                                "size": len(content)
                            }
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
        
        return {"status": "success", "processed_files": processed_files}
    
    def ingest_documentation(self, docs_path: str) -> Dict[str, Any]:
        """
        Ingest and process documentation
        Returns processed docs with embeddings and extracted structured data
        """
        self.logger.info(f"Processing documentation: {docs_path}")
        
        if COCOINDEX_AVAILABLE:
            return self._process_docs_with_cocoindex(docs_path)
        else:
            return self._process_docs_fallback(docs_path)
    
    def _process_docs_with_cocoindex(self, docs_path: str) -> Dict[str, Any]:
        """Process docs using CocoIndex"""
        # Define data source
        docs_source = LocalFile(path=docs_path, pattern="**/*.{md,txt,pdf}")
        
        # Create processing pipeline
        docs_flow = self.flow.import_from(docs_source)
        
        # Split docs into chunks
        doc_chunks = docs_flow.transform(
            SplitRecursively(
                language="markdown",
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )
        
        # Generate embeddings
        embedded_chunks = doc_chunks.transform(
            self.embedder
        )
        
        # Extract structured information using LLM
        extracted_data = doc_chunks.transform(
            ExtractByLlm(
                llm_spec=self.llm_spec,
                output_type={
                    "decisions": [{"summary": "str", "rationale": "str", "tags": "List[str]"}],
                    "progress": [{"description": "str", "status": "str"}],
                    "patterns": [{"name": "str", "description": "str"}]
                }
            )
        )
        
        # Export to vector store
        embedded_chunks.export_to(
            Qdrant(
                url=self.config.cocoindex.qdrant_url,
                collection_name="doc_chunks"
            )
        )
        
        # Export structured data to ConPort
        extracted_data.export_to(ConPortTarget(self.config.conport))
        
        # Execute the flow
        self.flow.run()
        
        return {"status": "success", "processed_files": len(docs_flow)}
    
    def _process_docs_fallback(self, docs_path: str) -> Dict[str, Any]:
        """Fallback implementation for doc processing"""
        processed_files = []
        
        for root, _, files in os.walk(docs_path):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Simple chunking
                        chunks = [content[i:i+self.config.chunk_size] 
                                 for i in range(0, len(content), self.config.chunk_size)]
                        
                        processed_files.append({
                            "file_path": file_path,
                            "chunks": chunks,
                            "metadata": {
                                "file_type": file.split('.')[-1],
                                "size": len(content)
                            }
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
        
        return {"status": "success", "processed_files": processed_files}
    
    def incremental_update(self, changed_files: List[str]) -> Dict[str, Any]:
        """
        Process incremental updates to the codebase or documentation
        """
        self.logger.info(f"Processing incremental updates for {len(changed_files)} files")
        
        # In a real implementation, this would use CocoIndex's incremental processing
        # For now, we'll reprocess the changed files
        results = {"updated_files": [], "errors": []}
        
        for file_path in changed_files:
            try:
                if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
                    # Process as code
                    result = self._process_single_code_file(file_path)
                    results["updated_files"].append(file_path)
                elif file_path.endswith(('.md', '.txt')):
                    # Process as documentation
                    result = self._process_single_doc_file(file_path)
                    results["updated_files"].append(file_path)
                else:
                    self.logger.warning(f"Unsupported file type: {file_path}")
            except Exception as e:
                self.logger.error(f"Error updating {file_path}: {e}")
                results["errors"].append({"file": file_path, "error": str(e)})
        
        return results
    
    def _process_single_code_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single code file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking
        chunks = [content[i:i+self.config.chunk_size] 
                 for i in range(0, len(content), self.config.chunk_size)]
        
        return {
            "file_path": file_path,
            "chunks": chunks,
            "metadata": {
                "file_type": file_path.split('.')[-1],
                "size": len(content)
            }
        }
    
    def _process_single_doc_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single documentation file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple chunking
        chunks = [content[i:i+self.config.chunk_size] 
                 for i in range(0, len(content), self.config.chunk_size)]
        
        return {
            "file_path": file_path,
            "chunks": chunks,
            "metadata": {
                "file_type": file_path.split('.')[-1],
                "size": len(content)
            }
        }

class ConPortTarget:
    """Custom target for exporting structured data to ConPort"""
    
    def __init__(self, config):
        self.config = config
        # In a real implementation, this would connect to ConPort's API
    
    def write(self, data):
        """Write extracted structured data to ConPort"""
        # This would use ConPort's MCP tools to log decisions, progress, etc.
        pass