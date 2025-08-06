import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CocoIndexConfig:
    """Configuration for CocoIndex data processing"""
    database_url: str = os.getenv("COCOINDEX_DATABASE_URL", "postgresql://localhost/cocoindex")
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    
@dataclass
class RepoHyperConfig:
    """Configuration for RepoHyper code intelligence"""
    model_path: str = os.getenv("REPOHYPER_MODEL_PATH", "./models/repo_hyper_model.pt")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qodo/Qodo-Embed-1-1.5B")
    
@dataclass
class ConPortConfig:
    """Configuration for Context Portal"""
    db_path: str = os.getenv("CONPORT_DB_PATH", "./context_portal/context.db")
    workspace_id: str = os.getenv("WORKSPACE_ID", "default_workspace")
    
@dataclass
class UnifiedConfig:
    """Unified configuration for the platform"""
    cocoindex: CocoIndexConfig = CocoIndexConfig()
    repohyper: RepoHyperConfig = RepoHyperConfig()
    conport: ConPortConfig = ConPortConfig()
    
    # Data sources
    code_repo_path: str = os.getenv("CODE_REPO_PATH", "./code_repo")
    docs_path: str = os.getenv("DOCS_PATH", "./docs")
    
    # LLM configuration
    llm_api_type: str = os.getenv("LLM_API_TYPE", "OPENAI")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4")
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    
    # Processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))