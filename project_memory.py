"""
Project Memory Layer (ConPort functionality)
Manages structured project knowledge and provides MCP interface
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Semantic search will be limited.")

from .config import UnifiedConfig

@dataclass
class Decision:
    """Represents a project decision"""
    id: str
    summary: str
    rationale: str
    tags: List[str]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Progress:
    """Represents project progress"""
    id: str
    description: str
    status: str  # TODO, IN_PROGRESS, DONE
    timestamp: str
    linked_item_type: Optional[str] = None
    linked_item_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemPattern:
    """Represents a system pattern"""
    id: str
    name: str
    description: str
    tags: List[str]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CustomData:
    """Represents custom project data"""
    id: str
    category: str
    key: str
    value: Any
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProductContext:
    """Represents product context"""
    id: str
    content: Dict[str, Any]
    timestamp: str
    version: int = 1

@dataclass
class ActiveContext:
    """Represents active context"""
    id: str
    content: Dict[str, Any]
    timestamp: str
    version: int = 1

class ProjectMemoryEngine:
    """Manages project memory using ConPort principles"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = config.conport.db_path
        self.workspace_id = config.conport.workspace_id
        self.embedding_model = None
        self._setup_database()
        self._setup_embedding_model()
    
    def _setup_database(self):
        """Initialize SQLite database"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                rationale TEXT NOT NULL,
                tags TEXT,  -- JSON array
                timestamp TEXT NOT NULL,
                metadata TEXT  -- JSON
            )
        ''')
        
        # Progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                linked_item_type TEXT,
                linked_item_id TEXT,
                metadata TEXT  -- JSON
            )
        ''')
        
        # System patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_patterns (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                tags TEXT,  -- JSON array
                timestamp TEXT NOT NULL,
                metadata TEXT  -- JSON
            )
        ''')
        
        # Custom data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_data (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,  -- JSON
                timestamp TEXT NOT NULL,
                metadata TEXT  -- JSON
            )
        ''')
        
        # Product context table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_context (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                content TEXT NOT NULL,  -- JSON
                timestamp TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            )
        ''')
        
        # Active context table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_context (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                content TEXT NOT NULL,  -- JSON
                timestamp TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1
            )
        ''')
        
        # Item links table (for knowledge graph)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_links (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_item_type TEXT NOT NULL,
                source_item_id TEXT NOT NULL,
                target_item_type TEXT NOT NULL,
                target_item_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                description TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # Create FTS tables for search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts 
            USING fts5(summary, rationale, tags, content=decisions)
        ''')
        
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS custom_data_fts 
            USING fts5(category, key, value, content=custom_data)
        ''')
        
        self.conn.commit()
    
    def _setup_embedding_model(self):
        """Setup embedding model for semantic search"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.config.repohyper.embedding_model)
            except Exception as e:
                self.logger.error(f"Error loading embedding model: {e}")
                self.embedding_model = None
    
    # Decision management
    def log_decision(self, summary: str, rationale: str, tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Log a new decision"""
        decision_id = f"decision_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO decisions (id, workspace_id, summary, rationale, tags, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision_id,
            self.workspace_id,
            summary,
            rationale,
            json.dumps(tags or []),
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        
        # Update FTS table
        cursor.execute('''
            INSERT INTO decisions_fts (rowid, summary, rationale, tags)
            VALUES (?, ?, ?, ?)
        ''', (
            cursor.lastrowid,
            summary,
            rationale,
            ' '.join(tags or [])
        ))
        
        self.conn.commit()
        return decision_id
    
    def get_decisions(self, limit: int = 10, tags_filter_include_all: List[str] = None, 
                     tags_filter_include_any: List[str] = None) -> List[Decision]:
        """Get decisions with optional tag filtering"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM decisions WHERE workspace_id = ?"
        params = [self.workspace_id]
        
        if tags_filter_include_all:
            placeholders = ','.join(['?'] * len(tags_filter_include_all))
            query += f" AND tags LIKE ? AND " + " AND ".join([f"tags LIKE ?"] * (len(tags_filter_include_all) - 1))
            params.extend([f'%{tag}%' for tag in tags_filter_include_all])
        
        if tags_filter_include_any:
            placeholders = ','.join(['?'] * len(tags_filter_include_any))
            query += f" AND (" + " OR ".join([f"tags LIKE ?"] * len(tags_filter_include_any)) + ")"
            params.extend([f'%{tag}%' for tag in tags_filter_include_any])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_decision(row) for row in rows]
    
    def search_decisions_fts(self, query_term: str, limit: int = 10) -> List[Decision]:
        """Search decisions using full-text search"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT d.* FROM decisions_fts f
            JOIN decisions d ON f.rowid = d.rowid
            WHERE f.decisions_fts MATCH ? AND d.workspace_id = ?
            ORDER BY rank
            LIMIT ?
        ''', (query_term, self.workspace_id, limit))
        
        rows = cursor.fetchall()
        return [self._row_to_decision(row) for row in rows]
    
    def delete_decision_by_id(self, decision_id: str) -> bool:
        """Delete a decision by ID"""
        cursor = self.conn.cursor()
        
        # Get rowid for FTS table
        cursor.execute("SELECT rowid FROM decisions WHERE id = ? AND workspace_id = ?", 
                      (decision_id, self.workspace_id))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        # Delete from main table
        cursor.execute("DELETE FROM decisions WHERE id = ? AND workspace_id = ?", 
                      (decision_id, self.workspace_id))
        
        # Delete from FTS table
        cursor.execute("DELETE FROM decisions_fts WHERE rowid = ?", (row['rowid'],))
        
        self.conn.commit()
        return True
    
    # Progress management
    def log_progress(self, description: str, status: str, linked_item_type: str = None, 
                   linked_item_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Log new progress"""
        progress_id = f"progress_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO progress (id, workspace_id, description, status, timestamp, linked_item_type, linked_item_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            progress_id,
            self.workspace_id,
            description,
            status,
            datetime.now().isoformat(),
            linked_item_type,
            linked_item_id,
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        return progress_id
    
    def get_progress(self, status_filter: str = None, parent_id_filter: str = None, limit: int = 10) -> List[Progress]:
        """Get progress entries with optional filtering"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM progress WHERE workspace_id = ?"
        params = [self.workspace_id]
        
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)
        
        if parent_id_filter:
            query += " AND linked_item_id = ?"
            params.append(parent_id_filter)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_progress(row) for row in rows]
    
    def update_progress(self, progress_id: str, status: str = None, description: str = None, 
                       parent_id: str = None) -> bool:
        """Update an existing progress entry"""
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status)
        
        if description:
            updates.append("description = ?")
            params.append(description)
        
        if parent_id:
            updates.append("linked_item_id = ?")
            params.append(parent_id)
        
        if not updates:
            return False
        
        query = f"UPDATE progress SET {', '.join(updates)} WHERE id = ? AND workspace_id = ?"
        params.extend([progress_id, self.workspace_id])
        
        cursor.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def delete_progress_by_id(self, progress_id: str) -> bool:
        """Delete a progress entry by ID"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM progress WHERE id = ? AND workspace_id = ?", 
                      (progress_id, self.workspace_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    # System pattern management
    def log_system_pattern(self, name: str, description: str, tags: List[str] = None, 
                          metadata: Dict[str, Any] = None) -> str:
        """Log a new system pattern"""
        pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO system_patterns (id, workspace_id, name, description, tags, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            self.workspace_id,
            name,
            description,
            json.dumps(tags or []),
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        return pattern_id
    
    def get_system_patterns(self, tags_filter_include_all: List[str] = None, 
                           tags_filter_include_any: List[str] = None, limit: int = 10) -> List[SystemPattern]:
        """Get system patterns with optional tag filtering"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM system_patterns WHERE workspace_id = ?"
        params = [self.workspace_id]
        
        if tags_filter_include_all:
            placeholders = ','.join(['?'] * len(tags_filter_include_all))
            query += f" AND tags LIKE ? AND " + " AND ".join([f"tags LIKE ?"] * (len(tags_filter_include_all) - 1))
            params.extend([f'%{tag}%' for tag in tags_filter_include_all])
        
        if tags_filter_include_any:
            placeholders = ','.join(['?'] * len(tags_filter_include_any))
            query += f" AND (" + " OR ".join([f"tags LIKE ?"] * len(tags_filter_include_any)) + ")"
            params.extend([f'%{tag}%' for tag in tags_filter_include_any])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_system_pattern(row) for row in rows]
    
    def delete_system_pattern_by_id(self, pattern_id: str) -> bool:
        """Delete a system pattern by ID"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM system_patterns WHERE id = ? AND workspace_id = ?", 
                      (pattern_id, self.workspace_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    # Custom data management
    def log_custom_data(self, category: str, key: str, value: Any, metadata: Dict[str, Any] = None) -> str:
        """Log custom data"""
        data_id = f"data_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO custom_data (id, workspace_id, category, key, value, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_id,
            self.workspace_id,
            category,
            key,
            json.dumps(value),
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        
        # Update FTS table
        cursor.execute('''
            INSERT INTO custom_data_fts (rowid, category, key, value)
            VALUES (?, ?, ?, ?)
        ''', (
            cursor.lastrowid,
            category,
            key,
            json.dumps(value)
        ))
        
        self.conn.commit()
        return data_id
    
    def get_custom_data(self, category: str = None, key: str = None) -> List[CustomData]:
        """Get custom data with optional filtering"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM custom_data WHERE workspace_id = ?"
        params = [self.workspace_id]
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if key:
            query += " AND key = ?"
            params.append(key)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_custom_data(row) for row in rows]
    
    def search_custom_data_value_fts(self, query_term: str, category_filter: str = None, limit: int = 10) -> List[CustomData]:
        """Search custom data using full-text search"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT c.* FROM custom_data_fts f
            JOIN custom_data c ON f.rowid = c.rowid
            WHERE f.custom_data_fts MATCH ? AND c.workspace_id = ?
        '''
        params = [query_term, self.workspace_id]
        
        if category_filter:
            query += " AND c.category = ?"
            params.append(category_filter)
        
        query += " ORDER BY rank LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [self._row_to_custom_data(row) for row in rows]
    
    def delete_custom_data(self, category: str, key: str) -> bool:
        """Delete custom data by category and key"""
        cursor = self.conn.cursor()
        
        # Get rowid for FTS table
        cursor.execute("SELECT rowid FROM custom_data WHERE category = ? AND key = ? AND workspace_id = ?", 
                      (category, key, self.workspace_id))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        # Delete from main table
        cursor.execute("DELETE FROM custom_data WHERE category = ? AND key = ? AND workspace_id = ?", 
                      (category, key, self.workspace_id))
        
        # Delete from FTS table
        cursor.execute("DELETE FROM custom_data_fts WHERE rowid = ?", (row['rowid'],))
        
        self.conn.commit()
        return True
    
    # Product and active context management
    def update_product_context(self, content: Dict[str, Any]) -> bool:
        """Update product context"""
        context_id = f"product_context_{self.workspace_id}"
        
        cursor = self.conn.cursor()
        
        # Check if context exists
        cursor.execute("SELECT version FROM product_context WHERE id = ? AND workspace_id = ?", 
                      (context_id, self.workspace_id))
        row = cursor.fetchone()
        
        if row:
            # Update existing context
            new_version = row['version'] + 1
            cursor.execute('''
                UPDATE product_context 
                SET content = ?, timestamp = ?, version = ?
                WHERE id = ? AND workspace_id = ?
            ''', (
                json.dumps(content),
                datetime.now().isoformat(),
                new_version,
                context_id,
                self.workspace_id
            ))
        else:
            # Create new context
            cursor.execute('''
                INSERT INTO product_context (id, workspace_id, content, timestamp, version)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                context_id,
                self.workspace_id,
                json.dumps(content),
                datetime.now().isoformat(),
                1
            ))
        
        self.conn.commit()
        return True
    
    def get_product_context(self) -> Optional[ProductContext]:
        """Get product context"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM product_context WHERE id = ? AND workspace_id = ?", 
                      (f"product_context_{self.workspace_id}", self.workspace_id))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_product_context(row)
        return None
    
    def update_active_context(self, content: Dict[str, Any]) -> bool:
        """Update active context"""
        context_id = f"active_context_{self.workspace_id}"
        
        cursor = self.conn.cursor()
        
        # Check if context exists
        cursor.execute("SELECT version FROM active_context WHERE id = ? AND workspace_id = ?", 
                      (context_id, self.workspace_id))
        row = cursor.fetchone()
        
        if row:
            # Update existing context
            new_version = row['version'] + 1
            cursor.execute('''
                UPDATE active_context 
                SET content = ?, timestamp = ?, version = ?
                WHERE id = ? AND workspace_id = ?
            ''', (
                json.dumps(content),
                datetime.now().isoformat(),
                new_version,
                context_id,
                self.workspace_id
            ))
        else:
            # Create new context
            cursor.execute('''
                INSERT INTO active_context (id, workspace_id, content, timestamp, version)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                context_id,
                self.workspace_id,
                json.dumps(content),
                datetime.now().isoformat(),
                1
            ))
        
        self.conn.commit()
        return True
    
    def get_active_context(self) -> Optional[ActiveContext]:
        """Get active context"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM active_context WHERE id = ? AND workspace_id = ?", 
                      (f"active_context_{self.workspace_id}", self.workspace_id))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_active_context(row)
        return None
    
    # Knowledge graph links
    def link_conport_items(self, source_item_type: str, source_item_id: str, target_item_type: str, 
                           target_item_id: str, relationship_type: str, description: str = None) -> str:
        """Create a link between two ConPort items"""
        link_id = f"link_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO item_links (id, workspace_id, source_item_type, source_item_id, 
                                   target_item_type, target_item_id, relationship_type, description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            link_id,
            self.workspace_id,
            source_item_type,
            source_item_id,
            target_item_type,
            target_item_id,
            relationship_type,
            description,
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
        return link_id
    
    def get_linked_items(self, item_type: str, item_id: str, relationship_type_filter: str = None, 
                        linked_item_type_filter: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get items linked to the specified item"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT * FROM item_links 
            WHERE workspace_id = ? 
            AND ((source_item_type = ? AND source_item_id = ?) OR (target_item_type = ? AND target_item_id = ?))
        '''
        params = [self.workspace_id, item_type, item_id, item_type, item_id]
        
        if relationship_type_filter:
            query += " AND relationship_type = ?"
            params.append(relationship_type_filter)
        
        if linked_item_type_filter:
            query += " AND (source_item_type = ? OR target_item_type = ?)"
            params.extend([linked_item_type_filter, linked_item_type_filter])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            # Determine which item is the linked one (not the input item)
            if row['source_item_type'] == item_type and row['source_item_id'] == item_id:
                linked_item_type = row['target_item_type']
                linked_item_id = row['target_item_id']
            else:
                linked_item_type = row['source_item_type']
                linked_item_id = row['source_item_id']
            
            results.append({
                "link_id": row['id'],
                "linked_item_type": linked_item_type,
                "linked_item_id": linked_item_id,
                "relationship_type": row['relationship_type'],
                "description": row['description'],
                "timestamp": row['timestamp']
            })
        
        return results
    
    # Semantic search
    def semantic_search_conport(self, query_text: str, top_k: int = 5, 
                              filter_item_types: List[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search across ConPort items"""
        if not self.embedding_model:
            self.logger.warning("Embedding model not available, falling back to keyword search")
            return self._keyword_search_conport(query_text, top_k, filter_item_types)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Get all searchable items
        items = []
        
        # Add decisions
        if not filter_item_types or "decision" in filter_item_types:
            decisions = self.get_decisions(limit=100)  # Get recent decisions
            for decision in decisions:
                items.append({
                    "type": "decision",
                    "id": decision.id,
                    "text": f"{decision.summary}: {decision.rationale}",
                    "tags": decision.tags,
                    "timestamp": decision.timestamp
                })
        
        # Add system patterns
        if not filter_item_types or "system_pattern" in filter_item_types:
            patterns = self.get_system_patterns(limit=100)
            for pattern in patterns:
                items.append({
                    "type": "system_pattern",
                    "id": pattern.id,
                    "text": f"{pattern.name}: {pattern.description}",
                    "tags": pattern.tags,
                    "timestamp": pattern.timestamp
                })
        
        # Add custom data
        if not filter_item_types or "custom_data" in filter_item_types:
            custom_data = self.get_custom_data()
            for data in custom_data:
                items.append({
                    "type": "custom_data",
                    "id": data.id,
                    "text": f"{data.category}.{data.key}: {json.dumps(data.value)}",
                    "category": data.category,
                    "timestamp": data.timestamp
                })
        
        # Calculate similarities
        results = []
        for item in items:
            item_embedding = self.embedding_model.encode(item['text']).tolist()
            similarity = self._calculate_cosine_similarity(query_embedding, item_embedding)
            
            results.append({
                "item": item,
                "score": similarity
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _keyword_search_conport(self, query_text: str, top_k: int = 5, 
                               filter_item_types: List[str] = None) -> List[Dict[str, Any]]:
        """Fallback keyword search for ConPort items"""
        query_lower = query_text.lower()
        results = []
        
        # Search decisions
        if not filter_item_types or "decision" in filter_item_types:
            decisions = self.search_decisions_fts(query_text, limit=top_k)
            for decision in decisions:
                results.append({
                    "item": {
                        "type": "decision",
                        "id": decision.id,
                        "text": f"{decision.summary}: {decision.rationale}",
                        "tags": decision.tags,
                        "timestamp": decision.timestamp
                    },
                    "score": 1.0  # Default score for keyword matches
                })
        
        # Search custom data
        if not filter_item_types or "custom_data" in filter_item_types:
            custom_data = self.search_custom_data_value_fts(query_text, limit=top_k)
            for data in custom_data:
                results.append({
                    "item": {
                        "type": "custom_data",
                        "id": data.id,
                        "text": f"{data.category}.{data.key}: {json.dumps(data.value)}",
                        "category": data.category,
                        "timestamp": data.timestamp
                    },
                    "score": 1.0
                })
        
        # Sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
        except ImportError:
            # Fallback implementation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
    
    # Helper methods
    def _row_to_decision(self, row) -> Decision:
        """Convert database row to Decision object"""
        return Decision(
            id=row['id'],
            summary=row['summary'],
            rationale=row['rationale'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            timestamp=row['timestamp'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    def _row_to_progress(self, row) -> Progress:
        """Convert database row to Progress object"""
        return Progress(
            id=row['id'],
            description=row['description'],
            status=row['status'],
            timestamp=row['timestamp'],
            linked_item_type=row['linked_item_type'],
            linked_item_id=row['linked_item_id'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    def _row_to_system_pattern(self, row) -> SystemPattern:
        """Convert database row to SystemPattern object"""
        return SystemPattern(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            timestamp=row['timestamp'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    def _row_to_custom_data(self, row) -> CustomData:
        """Convert database row to CustomData object"""
        return CustomData(
            id=row['id'],
            category=row['category'],
            key=row['key'],
            value=json.loads(row['value']),
            timestamp=row['timestamp'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
    
    def _row_to_product_context(self, row) -> ProductContext:
        """Convert database row to ProductContext object"""
        return ProductContext(
            id=row['id'],
            content=json.loads(row['content']),
            timestamp=row['timestamp'],
            version=row['version']
        )
    
    def _row_to_active_context(self, row) -> ActiveContext:
        """Convert database row to ActiveContext object"""
        return ActiveContext(
            id=row['id'],
            content=json.loads(row['content']),
            timestamp=row['timestamp'],
            version=row['version']
        )
    
    def get_recent_activity_summary(self, hours_ago: int = 24, since_timestamp: str = None, 
                                   limit_per_type: int = 3) -> Dict[str, List[Any]]:
        """Get summary of recent activity across all item types"""
        cursor = self.conn.cursor()
        
        # Calculate timestamp filter
        if since_timestamp:
            timestamp_filter = since_timestamp
        else:
            from datetime import datetime, timedelta
            timestamp_filter = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
        
        # Get recent decisions
        cursor.execute('''
            SELECT * FROM decisions 
            WHERE workspace_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.workspace_id, timestamp_filter, limit_per_type))
        recent_decisions = [self._row_to_decision(row) for row in cursor.fetchall()]
        
        # Get recent progress
        cursor.execute('''
            SELECT * FROM progress 
            WHERE workspace_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.workspace_id, timestamp_filter, limit_per_type))
        recent_progress = [self._row_to_progress(row) for row in cursor.fetchall()]
        
        # Get recent system patterns
        cursor.execute('''
            SELECT * FROM system_patterns 
            WHERE workspace_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.workspace_id, timestamp_filter, limit_per_type))
        recent_patterns = [self._row_to_system_pattern(row) for row in cursor.fetchall()]
        
        # Get recent custom data
        cursor.execute('''
            SELECT * FROM custom_data 
            WHERE workspace_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.workspace_id, timestamp_filter, limit_per_type))
        recent_custom_data = [self._row_to_custom_data(row) for row in cursor.fetchall()]
        
        return {
            "decisions": recent_decisions,
            "progress": recent_progress,
            "system_patterns": recent_patterns,
            "custom_data": recent_custom_data
        }