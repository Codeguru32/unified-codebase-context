"""
Main entry point for the Unified Codebase Context Platform
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from .config import UnifiedConfig
from .data_ingestion import DataIngestionEngine
from .code_intelligence import CodeIntelligenceEngine
from .project_memory import ProjectMemoryEngine
from .query_engine import QueryEngine
from .api import app

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('unified_platform.log')
        ]
    )

def initialize_workspace(config: UnifiedConfig):
    """Initialize the workspace with necessary directories and databases"""
    logger = logging.getLogger(__name__)
    
    # Create directories
    os.makedirs(os.path.dirname(config.conport.db_path), exist_ok=True)
    os.makedirs(config.code_repo_path, exist_ok=True)
    os.makedirs(config.docs_path, exist_ok=True)
    
    # Initialize project memory
    memory_engine = ProjectMemoryEngine(config)
    
    # Set default product context if not exists
    product_context = memory_engine.get_product_context()
    if not product_context:
        memory_engine.update_product_context({
            "project_name": "Untitled Project",
            "description": "A software project",
            "goals": [],
            "architecture": {},
            "key_features": []
        })
        logger.info("Initialized default product context")
    
    # Set default active context if not exists
    active_context = memory_engine.get_active_context()
    if not active_context:
        memory_engine.update_active_context({
            "current_focus": "Initial setup",
            "open_issues": [],
            "recent_changes": []
        })
        logger.info("Initialized default active context")
    
    logger.info("Workspace initialized successfully")

def ingest_data(config: UnifiedConfig):
    """Ingest initial data from code repository and documentation"""
    logger = logging.getLogger(__name__)
    
    # Initialize data ingestion engine
    ingestion_engine = DataIngestionEngine(config)
    
    # Ingest code repository
    if os.path.exists(config.code_repo_path):
        logger.info(f"Ingesting code repository from {config.code_repo_path}")
        result = ingestion_engine.ingest_code_repository(config.code_repo_path)
        logger.info(f"Code ingestion result: {result}")
    else:
        logger.warning(f"Code repository path does not exist: {config.code_repo_path}")
    
    # Ingest documentation
    if os.path.exists(config.docs_path):
        logger.info(f"Ingesting documentation from {config.docs_path}")
        result = ingestion_engine.ingest_documentation(config.docs_path)
        logger.info(f"Documentation ingestion result: {result}")
    else:
        logger.warning(f"Documentation path does not exist: {config.docs_path}")

def build_code_graph(config: UnifiedConfig):
    """Build the code property graph"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config.code_repo_path):
        logger.warning(f"Code repository path does not exist: {config.code_repo_path}")
        return
    
    logger.info("Building code property graph")
    code_engine = CodeIntelligenceEngine(config)
    cpg = code_engine.build_code_graph(config.code_repo_path)
    logger.info(f"Built code property graph with {len(cpg.entities)} entities and {len(cpg.relations)} relations")

def start_api_server(config: UnifiedConfig, host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    # Import here to avoid circular imports
    import uvicorn
    
    uvicorn.run(
        "unified_codebase_context.api:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Codebase Context Platform")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--init", action="store_true", help="Initialize workspace")
    parser.add_argument("--ingest", action="store_true", help="Ingest data from sources")
    parser.add_argument("--build-graph", action="store_true", help="Build code property graph")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = UnifiedConfig()
    if args.config:
        # Load from file if provided
        # This would implement loading from a config file
        pass
    
    logger.info("Starting Unified Codebase Context Platform")
    
    # Initialize workspace if requested
    if args.init:
        logger.info("Initializing workspace")
        initialize_workspace(config)
    
    # Ingest data if requested
    if args.ingest:
        logger.info("Ingesting data")
        ingest_data(config)
    
    # Build code graph if requested
    if args.build_graph:
        logger.info("Building code graph")
        build_code_graph(config)
    
    # Start API server if requested
    if args.serve:
        logger.info("Starting API server")
        start_api_server(config, args.host, args.port)
    
    # If no action specified, show help
    if not any([args.init, args.ingest, args.build_graph, args.serve]):
        parser.print_help()

if __name__ == "__main__":
    main()