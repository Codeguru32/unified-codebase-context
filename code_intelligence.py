"""
Code Intelligence Layer (RepoHyper functionality)
Handles code property graph construction and semantic code analysis
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import networkx as nx
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Using fallback implementation.")

from .config import UnifiedConfig

@dataclass
class CodeEntity:
    """Represents a code entity (function, class, etc.)"""
    id: str
    name: str
    type: str  # function, class, method, etc.
    file_path: str
    start_line: int
    end_line: int
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CodeRelation:
    """Represents a relationship between code entities"""
    source_id: str
    target_id: str
    relation_type: str  # calls, inherits, implements, etc.
    metadata: Optional[Dict[str, Any]] = None

class CodePropertyGraph:
    """Represents the code property graph (CPG)"""
    
    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.relations: List[CodeRelation] = []
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity: CodeEntity):
        """Add a code entity to the graph"""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.__dict__)
    
    def add_relation(self, relation: CodeRelation):
        """Add a relation between entities"""
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source_id, 
            relation.target_id,
            relation_type=relation.relation_type,
            **(relation.metadata or {})
        )
    
    def get_entity(self, entity_id: str) -> Optional[CodeEntity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)
    
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[CodeEntity]:
        """Get entities related to the given entity"""
        related = []
        
        if relation_type:
            # Get specific relation type
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if edge_data.get('relation_type') == relation_type:
                    related.append(self.entities.get(neighbor))
        else:
            # Get all related entities
            for neighbor in self.graph.neighbors(entity_id):
                related.append(self.entities.get(neighbor))
        
        return [r for r in related if r is not None]

class CodeIntelligenceEngine:
    """Handles code intelligence using RepoHyper principles"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cpg = CodePropertyGraph()
        self.embedding_model = None
        self.gnn_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize models for code analysis"""
        if TORCH_AVAILABLE:
            try:
                # Load embedding model
                self.embedding_model = SentenceTransformer(self.config.repohyper.embedding_model)
                
                # Load GNN model if available
                if os.path.exists(self.config.repohyper.model_path):
                    self.gnn_model = torch.load(self.config.repohyper.model_path)
                    self.gnn_model.eval()
            except Exception as e:
                self.logger.error(f"Error loading models: {e}")
                self.embedding_model = None
                self.gnn_model = None
    
    def build_code_graph(self, repo_path: str) -> CodePropertyGraph:
        """
        Build code property graph for the repository
        Returns the constructed graph
        """
        self.logger.info(f"Building code property graph for: {repo_path}")
        
        # Parse code files and extract entities
        entities = self._extract_code_entities(repo_path)
        
        # Add entities to CPG
        for entity in entities:
            self.cpg.add_entity(entity)
        
        # Extract relations between entities
        relations = self._extract_code_relations(repo_path, entities)
        
        # Add relations to CPG
        for relation in relations:
            self.cpg.add_relation(relation)
        
        # Generate embeddings for entities
        if self.embedding_model:
            self._generate_entity_embeddings()
        
        self.logger.info(f"Built CPG with {len(entities)} entities and {len(relations)} relations")
        return self.cpg
    
    def _extract_code_entities(self, repo_path: str) -> List[CodeEntity]:
        """Extract code entities from the repository"""
        entities = []
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.java')):
                    file_path = os.path.join(root, file)
                    try:
                        file_entities = self._parse_code_file(file_path)
                        entities.extend(file_entities)
                    except Exception as e:
                        self.logger.error(f"Error parsing {file_path}: {e}")
        
        return entities
    
    def _parse_code_file(self, file_path: str) -> List[CodeEntity]:
        """Parse a single code file and extract entities"""
        # This is a simplified implementation
        # In a real implementation, this would use Tree-sitter or similar
        
        entities = []
        file_type = file_path.split('.')[-1]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Simple regex-based extraction for demonstration
        import re
        
        if file_type == 'py':
            # Extract functions
            func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                start_line = content[:match.start()].count('\n') + 1
                entities.append(CodeEntity(
                    id=f"{file_path}:{func_name}",
                    name=func_name,
                    type="function",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,  # Simplified
                    content=lines[start_line-1] if start_line <= len(lines) else ""
                ))
            
            # Extract classes
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                start_line = content[:match.start()].count('\n') + 1
                entities.append(CodeEntity(
                    id=f"{file_path}:{class_name}",
                    name=class_name,
                    type="class",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=start_line,  # Simplified
                    content=lines[start_line-1] if start_line <= len(lines) else ""
                ))
        
        # Similar patterns can be added for other languages
        
        return entities
    
    def _extract_code_relations(self, repo_path: str, entities: List[CodeEntity]) -> List[CodeRelation]:
        """Extract relations between code entities"""
        relations = []
        
        # This is a simplified implementation
        # In a real implementation, this would use static analysis tools
        
        # For Python, we can look for function calls within other functions
        if any(e.file_path.endswith('.py') for e in entities):
            relations.extend(self._extract_python_relations(entities))
        
        return relations
    
    def _extract_python_relations(self, entities: List[CodeEntity]) -> List[CodeRelation]:
        """Extract relations for Python code"""
        relations = []
        
        # Group entities by file
        file_entities = {}
        for entity in entities:
            if entity.file_path not in file_entities:
                file_entities[entity.file_path] = []
            file_entities[entity.file_path].append(entity)
        
        # Analyze each file
        for file_path, file_ents in file_entities.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for function calls
                for func_entity in [e for e in file_ents if e.type == "function"]:
                    # Simple pattern to find function calls
                    func_calls = re.findall(rf'\b{func_entity.name}\s*\(', content)
                    if func_calls:
                        # Find functions that call this function
                        for caller in [e for e in file_ents if e.type == "function" and e.id != func_entity.id]:
                            caller_content = self._get_entity_content(caller, file_path)
                            if caller_content and func_entity.name in caller_content:
                                relations.append(CodeRelation(
                                    source_id=caller.id,
                                    target_id=func_entity.id,
                                    relation_type="calls"
                                ))
                
                # Look for class inheritance
                for class_entity in [e for e in file_ents if e.type == "class"]:
                    # Simple pattern to find inheritance
                    inherit_pattern = rf'class\s+\w+\s*\(\s*{class_entity.name}\s*\)'
                    if re.search(inherit_pattern, content):
                        for child_class in [e for e in file_ents if e.type == "class" and e.id != class_entity.id]:
                            relations.append(CodeRelation(
                                source_id=child_class.id,
                                target_id=class_entity.id,
                                relation_type="inherits"
                            ))
            
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return relations
    
    def _get_entity_content(self, entity: CodeEntity, file_path: str) -> Optional[str]:
        """Get the content of a code entity"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if entity.start_line <= len(lines):
                return lines[entity.start_line - 1]
            return None
        except Exception:
            return None
    
    def _generate_entity_embeddings(self):
        """Generate embeddings for code entities"""
        if not self.embedding_model:
            return
        
        for entity_id, entity in self.cpg.entities.items():
            try:
                # Create a text representation of the entity
                text = f"{entity.type} {entity.name}: {entity.content}"
                embedding = self.embedding_model.encode(text).tolist()
                entity.embedding = embedding
                
                # Update the graph node
                self.cpg.graph.nodes[entity_id]['embedding'] = embedding
            except Exception as e:
                self.logger.error(f"Error generating embedding for {entity_id}: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search over the codebase
        Returns relevant code entities with their scores
        """
        if not self.embedding_model:
            self.logger.warning("Embedding model not available, falling back to keyword search")
            return self._keyword_search(query, top_k)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Calculate similarity with all entities
        results = []
        for entity_id, entity in self.cpg.entities.items():
            if entity.embedding:
                similarity = self._calculate_cosine_similarity(query_embedding, entity.embedding)
                results.append({
                    "entity": entity,
                    "score": similarity,
                    "entity_id": entity_id
                })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not TORCH_AVAILABLE:
            # Fallback implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
        
        # Use PyTorch for better performance
        vec1_tensor = torch.tensor(vec1)
        vec2_tensor = torch.tensor(vec2)
        return torch.nn.functional.cosine_similarity(vec1_tensor, vec2_tensor, dim=0).item()
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword search implementation"""
        query_lower = query.lower()
        results = []
        
        for entity_id, entity in self.cpg.entities.items():
            score = 0
            # Check name match
            if query_lower in entity.name.lower():
                score += 1.0
            # Check content match
            if query_lower in entity.content.lower():
                score += 0.5
            
            if score > 0:
                results.append({
                    "entity": entity,
                    "score": score,
                    "entity_id": entity_id
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_context_for_completion(self, file_path: str, cursor_position: int, context_size: int = 5) -> Dict[str, Any]:
        """
        Get relevant context for code completion at a specific position
        Returns entities and relations that are relevant for completion
        """
        # Find entities in the same file
        file_entities = [e for e in self.cpg.entities.values() if e.file_path == file_path]
        
        # Find entities that are related to entities in this file
        related_entities = set()
        for entity in file_entities:
            related = self.cpg.get_related_entities(entity.id)
            related_entities.update(r.id for r in related)
        
        # Get the actual entities
        context_entities = []
        for entity_id in related_entities:
            entity = self.cpg.get_entity(entity_id)
            if entity:
                context_entities.append(entity)
        
        # Sort by proximity to cursor position (simplified)
        context_entities.sort(key=lambda e: abs(e.start_line - cursor_position))
        
        # Limit to context_size
        context_entities = context_entities[:context_size]
        
        return {
            "file_entities": file_entities,
            "related_entities": context_entities,
            "relations": [
                r for r in self.cpg.relations 
                if r.source_id in [e.id for e in file_entities] or 
                   r.target_id in [e.id for e in file_entities]
            ]
        }
    
    def update_graph(self, changed_files: List[str]):
        """Update the code property graph with changed files"""
        self.logger.info(f"Updating code graph with {len(changed_files)} changed files")
        
        # Remove entities and relations for changed files
        entities_to_remove = []
        for entity_id, entity in self.cpg.entities.items():
            if entity.file_path in changed_files:
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            # Remove entity
            if entity_id in self.cpg.entities:
                del self.cpg.entities[entity_id]
            
            # Remove from graph
            if entity_id in self.cpg.graph:
                self.cpg.graph.remove_node(entity_id)
        
        # Remove relations involving removed entities
        self.cpg.relations = [
            r for r in self.cpg.relations 
            if r.source_id not in entities_to_remove and r.target_id not in entities_to_remove
        ]
        
        # Re-parse changed files and add new entities
        for file_path in changed_files:
            try:
                new_entities = self._parse_code_file(file_path)
                for entity in new_entities:
                    self.cpg.add_entity(entity)
                
                # Extract new relations
                new_relations = self._extract_code_relations(file_path, new_entities)
                for relation in new_relations:
                    self.cpg.add_relation(relation)
                
                # Generate embeddings for new entities
                if self.embedding_model:
                    for entity in new_entities:
                        text = f"{entity.type} {entity.name}: {entity.content}"
                        entity.embedding = self.embedding_model.encode(text).tolist()
            except Exception as e:
                self.logger.error(f"Error updating {file_path}: {e}")
        
        self.logger.info("Code graph updated successfully")