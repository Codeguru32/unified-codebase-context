"""
API Interface for the Unified Codebase Context Platform
Provides REST API for external tools and AI assistants
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .config import UnifiedConfig
from .query_engine import QueryEngine, QueryResult, CodeCompletionContext
from .project_memory import Decision, Progress, SystemPattern, CustomData

# Pydantic models for API
class DecisionRequest(BaseModel):
    summary: str
    rationale: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProgressRequest(BaseModel):
    description: str
    status: str
    linked_item_type: Optional[str] = None
    linked_item_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PatternRequest(BaseModel):
    name: str
    description: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CustomDataRequest(BaseModel):
    category: str
    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ProductContextRequest(BaseModel):
    content: Dict[str, Any]

class ActiveContextRequest(BaseModel):
    content: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    search_code: bool = True
    search_docs: bool = True
    search_knowledge: bool = True
    top_k: int = 5

class CodeCompletionRequest(BaseModel):
    file_path: str
    cursor_position: int
    context_size: int = 5

class LinkItemsRequest(BaseModel):
    source_item_type: str
    source_item_id: str
    target_item_type: str
    target_item_id: str
    relationship_type: str
    description: Optional[str] = None

class UpdateRequest(BaseModel):
    changed_files: List[str]

# Global variables for the app
config = None
query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global config, query_engine
    config = UnifiedConfig()
    query_engine = QueryEngine(config)
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Unified Codebase Context Platform API",
    description="API for accessing unified codebase context and project knowledge",
    version="1.0.0",
    lifespan=lifespan
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Decision management endpoints
@app.post("/decisions", response_model=Dict[str, str])
async def log_decision(request: DecisionRequest):
    """Log a new decision"""
    try:
        decision_id = query_engine.memory_engine.log_decision(
            summary=request.summary,
            rationale=request.rationale,
            tags=request.tags,
            metadata=request.metadata
        )
        return {"decision_id": decision_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decisions", response_model=List[Dict[str, Any]])
async def get_decisions(
    limit: int = 10,
    tags_filter_include_all: Optional[str] = None,
    tags_filter_include_any: Optional[str] = None
):
    """Get decisions with optional tag filtering"""
    try:
        tags_all = tags_filter_include_all.split(',') if tags_filter_include_all else None
        tags_any = tags_filter_include_any.split(',') if tags_filter_include_any else None
        
        decisions = query_engine.memory_engine.get_decisions(
            limit=limit,
            tags_filter_include_all=tags_all,
            tags_filter_include_any=tags_any
        )
        
        return [
            {
                "id": d.id,
                "summary": d.summary,
                "rationale": d.rationale,
                "tags": d.tags,
                "timestamp": d.timestamp,
                "metadata": d.metadata
            }
            for d in decisions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/decisions/{decision_id}")
async def delete_decision(decision_id: str):
    """Delete a decision by ID"""
    try:
        success = query_engine.memory_engine.delete_decision_by_id(decision_id)
        if not success:
            raise HTTPException(status_code=404, detail="Decision not found")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Progress management endpoints
@app.post("/progress", response_model=Dict[str, str])
async def log_progress(request: ProgressRequest):
    """Log new progress"""
    try:
        progress_id = query_engine.memory_engine.log_progress(
            description=request.description,
            status=request.status,
            linked_item_type=request.linked_item_type,
            linked_item_id=request.linked_item_id,
            metadata=request.metadata
        )
        return {"progress_id": progress_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress", response_model=List[Dict[str, Any]])
async def get_progress(
    status_filter: Optional[str] = None,
    parent_id_filter: Optional[str] = None,
    limit: int = 10
):
    """Get progress entries with optional filtering"""
    try:
        progress_entries = query_engine.memory_engine.get_progress(
            status_filter=status_filter,
            parent_id_filter=parent_id_filter,
            limit=limit
        )
        
        return [
            {
                "id": p.id,
                "description": p.description,
                "status": p.status,
                "timestamp": p.timestamp,
                "linked_item_type": p.linked_item_type,
                "linked_item_id": p.linked_item_id,
                "metadata": p.metadata
            }
            for p in progress_entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/progress/{progress_id}")
async def update_progress(
    progress_id: str,
    status: Optional[str] = None,
    description: Optional[str] = None,
    parent_id: Optional[str] = None
):
    """Update an existing progress entry"""
    try:
        success = query_engine.memory_engine.update_progress(
            progress_id=progress_id,
            status=status,
            description=description,
            parent_id=parent_id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Progress entry not found")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/progress/{progress_id}")
async def delete_progress(progress_id: str):
    """Delete a progress entry by ID"""
    try:
        success = query_engine.memory_engine.delete_progress_by_id(progress_id)
        if not success:
            raise HTTPException(status_code=404, detail="Progress entry not found")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System pattern management endpoints
@app.post("/patterns", response_model=Dict[str, str])
async def log_system_pattern(request: PatternRequest):
    """Log a new system pattern"""
    try:
        pattern_id = query_engine.memory_engine.log_system_pattern(
            name=request.name,
            description=request.description,
            tags=request.tags,
            metadata=request.metadata
        )
        return {"pattern_id": pattern_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns", response_model=List[Dict[str, Any]])
async def get_system_patterns(
    tags_filter_include_all: Optional[str] = None,
    tags_filter_include_any: Optional[str] = None,
    limit: int = 10
):
    """Get system patterns with optional tag filtering"""
    try:
        tags_all = tags_filter_include_all.split(',') if tags_filter_include_all else None
        tags_any = tags_filter_include_any.split(',') if tags_filter_include_any else None
        
        patterns = query_engine.memory_engine.get_system_patterns(
            tags_filter_include_all=tags_all,
            tags_filter_include_any=tags_any,
            limit=limit
        )
        
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "tags": p.tags,
                "timestamp": p.timestamp,
                "metadata": p.metadata
            }
            for p in patterns
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patterns/{pattern_id}")
async def delete_system_pattern(pattern_id: str):
    """Delete a system pattern by ID"""
    try:
        success = query_engine.memory_engine.delete_system_pattern_by_id(pattern_id)
        if not success:
            raise HTTPException(status_code=404, detail="Pattern not found")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Custom data management endpoints
@app.post("/custom-data", response_model=Dict[str, str])
async def log_custom_data(request: CustomDataRequest):
    """Log custom data"""
    try:
        data_id = query_engine.memory_engine.log_custom_data(
            category=request.category,
            key=request.key,
            value=request.value,
            metadata=request.metadata
        )
        return {"data_id": data_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/custom-data", response_model=List[Dict[str, Any]])
async def get_custom_data(
    category: Optional[str] = None,
    key: Optional[str] = None
):
    """Get custom data with optional filtering"""
    try:
        custom_data = query_engine.memory_engine.get_custom_data(
            category=category,
            key=key
        )
        
        return [
            {
                "id": d.id,
                "category": d.category,
                "key": d.key,
                "value": d.value,
                "timestamp": d.timestamp,
                "metadata": d.metadata
            }
            for d in custom_data
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/custom-data/{category}/{key}")
async def delete_custom_data(category: str, key: str):
    """Delete custom data by category and key"""
    try:
        success = query_engine.memory_engine.delete_custom_data(category, key)
        if not success:
            raise HTTPException(status_code=404, detail="Custom data not found")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Product and active context endpoints
@app.put("/product-context")
async def update_product_context(request: ProductContextRequest):
    """Update product context"""
    try:
        success = query_engine.memory_engine.update_product_context(request.content)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update product context")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product-context")
async def get_product_context():
    """Get product context"""
    try:
        context = query_engine.memory_engine.get_product_context()
        if not context:
            return {"content": {}}
        return {
            "content": context.content,
            "timestamp": context.timestamp,
            "version": context.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/active-context")
async def update_active_context(request: ActiveContextRequest):
    """Update active context"""
    try:
        success = query_engine.memory_engine.update_active_context(request.content)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update active context")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/active-context")
async def get_active_context():
    """Get active context"""
    try:
        context = query_engine.memory_engine.get_active_context()
        if not context:
            return {"content": {}}
        return {
            "content": context.content,
            "timestamp": context.timestamp,
            "version": context.version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge graph endpoints
@app.post("/links", response_model=Dict[str, str])
async def link_items(request: LinkItemsRequest):
    """Create a link between two ConPort items"""
    try:
        link_id = query_engine.memory_engine.link_conport_items(
            source_item_type=request.source_item_type,
            source_item_id=request.source_item_id,
            target_item_type=request.target_item_type,
            target_item_id=request.target_item_id,
            relationship_type=request.relationship_type,
            description=request.description
        )
        return {"link_id": link_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/links/{item_type}/{item_id}")
async def get_linked_items(
    item_type: str,
    item_id: str,
    relationship_type_filter: Optional[str] = None,
    linked_item_type_filter: Optional[str] = None,
    limit: int = 10
):
    """Get items linked to the specified item"""
    try:
        linked_items = query_engine.memory_engine.get_linked_items(
            item_type=item_type,
            item_id=item_id,
            relationship_type_filter=relationship_type_filter,
            linked_item_type_filter=linked_item_type_filter,
            limit=limit
        )
        return {"linked_items": linked_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Search and query endpoints
@app.post("/search", response_model=List[Dict[str, Any]])
async def unified_search(request: SearchRequest):
    """Perform unified search across code, docs, and knowledge"""
    try:
        results = query_engine.unified_search(
            query=request.query,
            search_code=request.search_code,
            search_docs=request.search_docs,
            search_knowledge=request.search_knowledge,
            top_k=request.top_k
        )
        
        return [
            {
                "content": result.content,
                "source_type": result.source_type,
                "source_id": result.source_id,
                "metadata": result.metadata,
                "score": result.score
            }
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code-completion-context", response_model=Dict[str, Any])
async def get_code_completion_context(request: CodeCompletionRequest):
    """Get context for code completion"""
    try:
        context = query_engine.get_code_completion_context(
            file_path=request.file_path,
            cursor_position=request.cursor_position,
            context_size=request.context_size
        )
        
        return {
            "file_path": context.file_path,
            "cursor_position": context.cursor_position,
            "surrounding_code": context.surrounding_code,
            "relevant_entities": context.relevant_entities,
            "relevant_relations": context.relevant_relations,
            "project_context": context.project_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-context")
async def get_rag_context(query: str, max_tokens: int = 2000):
    """Get RAG context for LLM prompts"""
    try:
        context = query_engine.get_rag_context(query, max_tokens)
        return {"context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{entity_id}")
async def explain_code_entity(entity_id: str):
    """Explain a code entity with its relationships and relevant knowledge"""
    try:
        explanation = query_engine.explain_code_entity(entity_id)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decision-implementations/{decision_id}")
async def find_implementation_of_decision(decision_id: str):
    """Find code entities that implement a specific decision"""
    try:
        implementations = query_engine.find_implementation_of_decision(decision_id)
        return {"implementations": implementations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System update endpoints
@app.post("/update")
async def update_with_changes(request: UpdateRequest):
    """Update the system with changed files"""
    try:
        query_engine.update_with_changes(request.changed_files)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent-activity")
async def get_recent_activity(
    hours_ago: int = 24,
    since_timestamp: Optional[str] = None,
    limit_per_type: int = 3
):
    """Get summary of recent activity"""
    try:
        activity = query_engine.memory_engine.get_recent_activity_summary(
            hours_ago=hours_ago,
            since_timestamp=since_timestamp,
            limit_per_type=limit_per_type
        )
        return activity
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)