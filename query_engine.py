"""
Unified Query & RAG Engine
Integrates code intelligence and project memory for comprehensive context retrieval
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from .config import UnifiedConfig
from .code_intelligence import CodeIntelligenceEngine
from .project_memory import ProjectMemoryEngine

@dataclass
class QueryResult:
    """Represents a query result"""
    content: str
    source_type: str  # "code", "decision", "progress", "pattern", "custom_data", "doc"
    source_id: str
    metadata: Dict[str, Any]
    score: float = 0.0
    context: Optional[str] = None

@dataclass
class CodeCompletionContext:
    """Context for code completion"""
    file_path: str
    cursor_position: int
    surrounding_code: str
    relevant_entities: List[Dict[str, Any]]
    relevant_relations: List[Dict[str, Any]]
    project_context: Dict[str, Any]

class QueryEngine:
    """Unified query engine for retrieving context from code and project memory"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize component engines
        self.code_engine = CodeIntelligenceEngine(config)
        self.memory_engine = ProjectMemoryEngine(config)
        
        # Build initial code graph
        if config.code_repo_path:
            self.code_engine.build_code_graph(config.code_repo_path)
    
    def unified_search(self, query: str, search_code: bool = True, search_docs: bool = True, 
                      search_knowledge: bool = True, top_k: int = 5) -> List[QueryResult]:
        """
        Perform a unified search across code, documentation, and project knowledge
        Returns ranked list of relevant results
        """
        self.logger.info(f"Performing unified search for: {query}")
        
        all_results = []
        
        # Search code
        if search_code:
            code_results = self._search_code(query, top_k)
            all_results.extend(code_results)
        
        # Search documentation (would integrate with CocoIndex in full implementation)
        if search_docs:
            doc_results = self._search_docs(query, top_k)
            all_results.extend(doc_results)
        
        # Search project knowledge
        if search_knowledge:
            knowledge_results = self._search_knowledge(query, top_k)
            all_results.extend(knowledge_results)
        
        # Sort all results by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k results
        return all_results[:top_k]
    
    def _search_code(self, query: str, top_k: int) -> List[QueryResult]:
        """Search code entities"""
        results = []
        
        # Use semantic search from code engine
        code_matches = self.code_engine.semantic_search(query, top_k)
        
        for match in code_matches:
            entity = match["entity"]
            results.append(QueryResult(
                content=f"{entity.type} {entity.name}: {entity.content}",
                source_type="code",
                source_id=entity.id,
                metadata={
                    "file_path": entity.file_path,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "entity_type": entity.type
                },
                score=match["score"]
            ))
        
        return results
    
    def _search_docs(self, query: str, top_k: int) -> List[QueryResult]:
        """Search documentation (placeholder implementation)"""
        # In a full implementation, this would use CocoIndex's vector store
        # For now, return empty results
        return []
    
    def _search_knowledge(self, query: str, top_k: int) -> List[QueryResult]:
        """Search project knowledge"""
        results = []
        
        # Use semantic search from memory engine
        knowledge_matches = self.memory_engine.semantic_search_conport(query, top_k)
        
        for match in knowledge_matches:
            item = match["item"]
            results.append(QueryResult(
                content=item["text"],
                source_type=item["type"],
                source_id=item["id"],
                metadata={
                    "tags": item.get("tags", []),
                    "category": item.get("category"),
                    "timestamp": item.get("timestamp")
                },
                score=match["score"]
            ))
        
        return results
    
    def get_code_completion_context(self, file_path: str, cursor_position: int, 
                                  context_size: int = 5) -> CodeCompletionContext:
        """
        Get comprehensive context for code completion at a specific position
        Includes code entities, project knowledge, and documentation
        """
        # Get code context from code engine
        code_context = self.code_engine.get_context_for_completion(file_path, cursor_position, context_size)
        
        # Get surrounding code
        surrounding_code = self._get_surrounding_code(file_path, cursor_position)
        
        # Get relevant project knowledge
        project_context = self._get_relevant_project_context(file_path, cursor_position)
        
        return CodeCompletionContext(
            file_path=file_path,
            cursor_position=cursor_position,
            surrounding_code=surrounding_code,
            relevant_entities=[
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "file_path": entity.file_path,
                    "content": entity.content
                }
                for entity in code_context["file_entities"]
            ],
            relevant_relations=[
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relation_type
                }
                for rel in code_context["relations"]
            ],
            project_context=project_context
        )
    
    def _get_surrounding_code(self, file_path: str, cursor_position: int, window_size: int = 10) -> str:
        """Get surrounding code around cursor position"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = max(0, cursor_position - window_size)
            end_line = min(len(lines), cursor_position + window_size)
            
            surrounding_lines = lines[start_line:end_line]
            return ''.join(surrounding_lines)
        except Exception as e:
            self.logger.error(f"Error getting surrounding code: {e}")
            return ""
    
    def _get_relevant_project_context(self, file_path: str, cursor_position: int) -> Dict[str, Any]:
        """Get project knowledge relevant to the current context"""
        # Get file name to use as context
        file_name = file_path.split('/')[-1].split('.')[0]
        
        # Search for relevant decisions, patterns, etc.
        context = {}
        
        # Get product context
        product_context = self.memory_engine.get_product_context()
        if product_context:
            context["product"] = product_context.content
        
        # Get active context
        active_context = self.memory_engine.get_active_context()
        if active_context:
            context["active"] = active_context.content
        
        # Search for relevant decisions
        decisions = self.memory_engine.search_decisions_fts(file_name, limit=3)
        if decisions:
            context["relevant_decisions"] = [
                {
                    "summary": d.summary,
                    "rationale": d.rationale,
                    "tags": d.tags
                }
                for d in decisions
            ]
        
        # Search for relevant system patterns
        patterns = self.memory_engine.get_system_patterns(limit=3)
        if patterns:
            context["relevant_patterns"] = [
                {
                    "name": p.name,
                    "description": p.description,
                    "tags": p.tags
                }
                for p in patterns
            ]
        
        return context
    
    def get_rag_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        Generate RAG context for LLM prompts
        Combines relevant code, documentation, and project knowledge
        """
        # Get unified search results
        results = self.unified_search(query, top_k=10)
        
        # Format results as context
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Estimate token count (rough approximation)
            result_tokens = len(result.content.split()) * 1.3  # Rough estimate
            
            if current_tokens + result_tokens > max_tokens:
                break
            
            # Format the result
            formatted_result = f"--- {result.source_type.upper()} ---\n"
            formatted_result += f"Source: {result.source_id}\n"
            formatted_result += f"Content: {result.content}\n"
            
            # Add metadata if relevant
            if result.metadata.get("file_path"):
                formatted_result += f"File: {result.metadata['file_path']}\n"
            
            if result.metadata.get("tags"):
                formatted_result += f"Tags: {', '.join(result.metadata['tags'])}\n"
            
            context_parts.append(formatted_result)
            current_tokens += result_tokens
        
        return "\n".join(context_parts)
    
    def explain_code_entity(self, entity_id: str) -> Dict[str, Any]:
        """
        Explain a code entity with its relationships and relevant project knowledge
        """
        # Get the entity from code engine
        entity = self.code_engine.cpg.get_entity(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        # Get related entities
        related_entities = self.code_engine.cpg.get_related_entities(entity_id)
        
        # Get related project knowledge
        # Search for decisions or patterns related to this entity
        knowledge_results = self.memory_engine.semantic_search_conport(
            entity.name, top_k=3, filter_item_types=["decision", "system_pattern"]
        )
        
        # Format the explanation
        explanation = {
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "file_path": entity.file_path,
                "content": entity.content
            },
            "related_entities": [
                {
                    "id": rel.id,
                    "name": rel.name,
                    "type": rel.type,
                    "file_path": rel.file_path
                }
                for rel in related_entities
            ],
            "related_knowledge": [
                {
                    "type": item["item"]["type"],
                    "content": item["item"]["text"],
                    "score": item["score"]
                }
                for item in knowledge_results
            ]
        }
        
        return explanation
    
    def find_implementation_of_decision(self, decision_id: str) -> List[Dict[str, Any]]:
        """
        Find code entities that implement a specific decision
        """
        # Get the decision
        decisions = self.memory_engine.get_decisions()
        decision = next((d for d in decisions if d.id == decision_id), None)
        if not decision:
            return []
        
        # Search for code entities related to the decision
        # This would use more sophisticated linking in a real implementation
        # For now, we'll use semantic search
        code_results = self.code_engine.semantic_search(decision.summary, top_k=10)
        
        implementations = []
        for result in code_results:
            entity = result["entity"]
            implementations.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.type,
                "file_path": entity.file_path,
                "relevance_score": result["score"]
            })
        
        return implementations
    
    def update_with_changes(self, changed_files: List[str]):
        """
        Update the system with changed files
        """
        self.logger.info(f"Updating system with {len(changed_files)} changed files")
        
        # Update code graph
        self.code_engine.update_graph(changed_files)
        
        # In a full implementation, we would also:
        # 1. Process documentation changes with CocoIndex
        # 2. Extract new project knowledge from changed files
        # 3. Update embeddings and indexes