"""
Vector Database Service
Implements vector storage and retrieval using FAISS (local, free) with Pinecone fallback.
Supports Google Cloud Storage for persistence in cloud environments.

This is the core of character memory persistence:
- Semantic vectors: WHO the character IS (identity embeddings)
- Episodic vectors: WHAT the character HAS DONE (scene state embeddings)

Memory Architecture:
┌─────────────────────────────────────────────────┐
│              VECTOR DATABASE                     │
├─────────────────────┬───────────────────────────┤
│  SEMANTIC NAMESPACE │   EPISODIC NAMESPACE      │
├─────────────────────┼───────────────────────────┤
│  sem_char_abc123    │  epi_char_abc123_1        │
│  - face_embedding   │  - scene_1_embedding      │
│  - style_embedding  │  - tags, state metadata   │
│                     │  epi_char_abc123_2        │
│                     │  - scene_2_embedding      │
└─────────────────────┴───────────────────────────┘
"""

import os
import json
import pickle
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from app.core.config import settings

# Try to import FAISS, fall back to simple numpy-based similarity
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS not available, using NumPy-based similarity search")

# Try to import Pinecone for cloud option
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class LocalVectorStore:
    """
    Local vector store using FAISS or NumPy fallback.
    Persists to disk or Google Cloud Storage for durability.
    """
    
    def __init__(self, dimension: int = 512, storage_path: str = "./vector_store"):
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.use_gcs = settings.USE_GCS
        
        # GCS setup
        if self.use_gcs:
            from google.cloud import storage
            self.gcs_client = storage.Client(project=settings.GCP_PROJECT_ID)
            self.gcs_bucket = self.gcs_client.bucket(settings.GCS_BUCKET_OUTPUTS)
            self.gcs_prefix = "vectordb/"
            print(f"[VectorDB] Using GCS: {settings.GCS_BUCKET_OUTPUTS}/{self.gcs_prefix}")
        else:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            print(f"[VectorDB] Using local storage: {self.storage_path}")
        
        # Separate indices for semantic and episodic
        self.semantic_vectors: Dict[str, np.ndarray] = {}
        self.episodic_vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # FAISS index for fast similarity search
        self.faiss_index = None
        self.faiss_ids: List[str] = []
        
        # Load existing data
        self._load()
        
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
    
    def _load(self):
        """Load vectors from disk or GCS."""
        try:
            if self.use_gcs:
                self._load_from_gcs()
            else:
                self._load_from_disk()
        except Exception as e:
            print(f"[VectorDB] Error loading data: {e}")
    
    def _load_from_disk(self):
        """Load from local filesystem."""
        semantic_path = self.storage_path / "semantic.pkl"
        episodic_path = self.storage_path / "episodic.pkl"
        metadata_path = self.storage_path / "metadata.json"
        
        if semantic_path.exists():
            with open(semantic_path, 'rb') as f:
                self.semantic_vectors = pickle.load(f)
        
        if episodic_path.exists():
            with open(episodic_path, 'rb') as f:
                self.episodic_vectors = pickle.load(f)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _load_from_gcs(self):
        """Load from Google Cloud Storage."""
        for file_name, attr_name in [
            ("semantic.pkl", "semantic_vectors"),
            ("episodic.pkl", "episodic_vectors"),
            ("metadata.json", "metadata")
        ]:
            blob = self.gcs_bucket.blob(f"{self.gcs_prefix}{file_name}")
            if blob.exists():
                data = blob.download_as_bytes()
                if file_name.endswith(".pkl"):
                    setattr(self, attr_name, pickle.loads(data))
                else:
                    setattr(self, attr_name, json.loads(data))
    
    def _save(self):
        """Persist vectors to disk or GCS."""
        if self.use_gcs:
            self._save_to_gcs()
        else:
            self._save_to_disk()
    
    def _save_to_disk(self):
        """Save to local filesystem."""
        with open(self.storage_path / "semantic.pkl", 'wb') as f:
            pickle.dump(self.semantic_vectors, f)
        
        with open(self.storage_path / "episodic.pkl", 'wb') as f:
            pickle.dump(self.episodic_vectors, f)
        
        with open(self.storage_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _save_to_gcs(self):
        """Save to Google Cloud Storage."""
        # Save semantic vectors
        blob = self.gcs_bucket.blob(f"{self.gcs_prefix}semantic.pkl")
        blob.upload_from_string(pickle.dumps(self.semantic_vectors))
        
        # Save episodic vectors
        blob = self.gcs_bucket.blob(f"{self.gcs_prefix}episodic.pkl")
        blob.upload_from_string(pickle.dumps(self.episodic_vectors))
        
        # Save metadata
        blob = self.gcs_bucket.blob(f"{self.gcs_prefix}metadata.json")
        blob.upload_from_string(json.dumps(self.metadata, indent=2, default=str))
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from all vectors."""
        if not FAISS_AVAILABLE:
            return
        
        all_vectors = {**self.semantic_vectors, **self.episodic_vectors}
        if not all_vectors:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity for normalized vectors)
            self.faiss_ids = []
            return
        
        self.faiss_ids = list(all_vectors.keys())
        vectors = np.array([all_vectors[vid] for vid in self.faiss_ids]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(vectors)
    
    def upsert(self, vector_id: str, vector: np.ndarray, metadata: Dict = None, namespace: str = "semantic"):
        """Insert or update a vector."""
        vector = vector.astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        if namespace == "semantic":
            self.semantic_vectors[vector_id] = vector
        else:
            self.episodic_vectors[vector_id] = vector
        
        if metadata:
            self.metadata[vector_id] = metadata
        
        self._save()
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
        
        return vector_id
    
    def query(self, vector: np.ndarray, top_k: int = 5, namespace: str = None, filter_prefix: str = None) -> List[Dict]:
        """Query for similar vectors."""
        vector = vector.astype(np.float32)
        
        # Normalize query
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Select namespace
        if namespace == "semantic":
            candidates = self.semantic_vectors
        elif namespace == "episodic":
            candidates = self.episodic_vectors
        else:
            candidates = {**self.semantic_vectors, **self.episodic_vectors}
        
        # Filter by prefix if specified
        if filter_prefix:
            candidates = {k: v for k, v in candidates.items() if k.startswith(filter_prefix)}
        
        if not candidates:
            return []
        
        # Compute similarities
        results = []
        for vid, vec in candidates.items():
            similarity = float(np.dot(vector, vec))
            results.append({
                "id": vid,
                "score": similarity,
                "metadata": self.metadata.get(vid, {})
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get(self, vector_id: str) -> Optional[np.ndarray]:
        """Get a specific vector by ID."""
        if vector_id in self.semantic_vectors:
            return self.semantic_vectors[vector_id]
        if vector_id in self.episodic_vectors:
            return self.episodic_vectors[vector_id]
        return None
    
    def delete(self, vector_id: str):
        """Delete a vector."""
        if vector_id in self.semantic_vectors:
            del self.semantic_vectors[vector_id]
        if vector_id in self.episodic_vectors:
            del self.episodic_vectors[vector_id]
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        
        self._save()
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()
    
    def delete_by_prefix(self, prefix: str):
        """Delete all vectors matching a prefix."""
        to_delete = [k for k in self.semantic_vectors if k.startswith(prefix)]
        for k in to_delete:
            del self.semantic_vectors[k]
        
        to_delete = [k for k in self.episodic_vectors if k.startswith(prefix)]
        for k in to_delete:
            del self.episodic_vectors[k]
        
        to_delete = [k for k in self.metadata if k.startswith(prefix)]
        for k in to_delete:
            del self.metadata[k]
        
        self._save()
        if FAISS_AVAILABLE:
            self._rebuild_faiss_index()


class VectorDBService:
    """
    High-level service for vector database operations.
    Wraps local FAISS store with option to use Pinecone for production.
    """
    
    def __init__(self):
        self.dimension = 512  # ArcFace dimension, adjust as needed
        self.use_pinecone = bool(settings.PINECONE_API_KEY) and PINECONE_AVAILABLE
        
        if self.use_pinecone:
            self._init_pinecone()
        else:
            self.local_store = LocalVectorStore(
                dimension=self.dimension,
                storage_path=os.path.join(settings.LOCAL_STORAGE_PATH, "vectors")
            )
            print("[OK] VectorDB: Using local FAISS/NumPy store")
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            index_name = settings.PINECONE_INDEX
            
            # Check if index exists
            existing = [idx.name for idx in self.pc.list_indexes()]
            
            if index_name not in existing:
                # Create index
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENV
                    )
                )
                print(f"[OK] Created Pinecone index: {index_name}")
            
            self.index = self.pc.Index(index_name)
            print(f"[OK] VectorDB: Connected to Pinecone index '{index_name}'")
        except Exception as e:
            print(f"[WARNING] Pinecone init failed, falling back to local: {e}")
            self.use_pinecone = False
            self.local_store = LocalVectorStore(
                dimension=self.dimension,
                storage_path=os.path.join(settings.LOCAL_STORAGE_PATH, "vectors")
            )
    
    async def upsert_semantic(
        self, 
        character_id: str, 
        embedding: np.ndarray, 
        metadata: Dict = None
    ) -> str:
        """
        Upsert semantic (identity) embedding for a character.
        This is the character's permanent identity vector.
        """
        vector_id = f"sem_{character_id}"
        
        # Convert numpy types to native Python types for Pinecone compatibility
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        meta = {
            "character_id": character_id,
            "type": "semantic",
            **(convert_numpy_types(metadata) if metadata else {})
        }
        
        if self.use_pinecone:
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": meta
                }],
                namespace="semantic"
            )
        else:
            self.local_store.upsert(vector_id, embedding, meta, namespace="semantic")
        
        print(f"[OK] Upserted semantic vector: {vector_id}")
        return vector_id
    
    async def upsert_episodic(
        self, 
        character_id: str, 
        scene_index: int, 
        embedding: np.ndarray, 
        metadata: Dict = None
    ) -> str:
        """
        Upsert episodic (scene state) embedding.
        Each scene generates a new episodic memory.
        """
        vector_id = f"epi_{character_id}_{scene_index}"
        
        # Convert numpy types to native Python types for Pinecone compatibility
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        meta = {
            "character_id": character_id,
            "scene_index": scene_index,
            "type": "episodic",
            **(convert_numpy_types(metadata) if metadata else {})
        }
        
        if self.use_pinecone:
            self.index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": meta
                }],
                namespace="episodic"
            )
        else:
            self.local_store.upsert(vector_id, embedding, meta, namespace="episodic")
        
        print(f"[OK] Upserted episodic vector: {vector_id}")
        return vector_id
    
    async def query_semantic(self, character_id: str) -> Optional[np.ndarray]:
        """
        Retrieve the semantic (identity) embedding for a character.
        """
        vector_id = f"sem_{character_id}"
        
        if self.use_pinecone:
            result = self.index.fetch(ids=[vector_id], namespace="semantic")
            if result and vector_id in result.vectors:
                return np.array(result.vectors[vector_id].values)
            return None
        else:
            return self.local_store.get(vector_id)
    
    async def query_episodic(
        self, 
        character_id: str, 
        top_k: int = None
    ) -> List[Dict]:
        """
        Retrieve top-K episodic embeddings for a character.
        Returns most recent scenes by default.
        """
        top_k = top_k or settings.EPISODIC_TOP_K
        prefix = f"epi_{character_id}_"
        
        if self.use_pinecone:
            # For Pinecone, we need a query vector to search
            # Use the semantic vector as the query
            semantic = await self.query_semantic(character_id)
            if semantic is None:
                return []
            
            results = self.index.query(
                vector=semantic.tolist(),
                top_k=top_k,
                namespace="episodic",
                include_metadata=True,
                filter={"character_id": character_id}
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        else:
            # For local, get all episodic for this character
            results = []
            for vid, vec in self.local_store.episodic_vectors.items():
                if vid.startswith(prefix):
                    results.append({
                        "id": vid,
                        "vector": vec,
                        "metadata": self.local_store.metadata.get(vid, {})
                    })
            
            # Sort by scene index descending (most recent first)
            results.sort(
                key=lambda x: x["metadata"].get("scene_index", 0), 
                reverse=True
            )
            return results[:top_k]
    
    async def query_similar(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5, 
        namespace: str = None
    ) -> List[Dict]:
        """
        Query for similar vectors using cosine similarity.
        """
        if self.use_pinecone:
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        else:
            return self.local_store.query(query_vector, top_k, namespace)
    
    async def merge_embeddings(
        self, 
        semantic: np.ndarray, 
        episodic_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Merge semantic and episodic embeddings using weighted decay.
        
        Algorithm:
        merged = w_s * semantic + Σ (w_e * decay^i * episodic_i)
        
        Where:
        - w_s = semantic weight (default 0.6)
        - w_e = episodic weight (default 0.4)
        - decay = temporal decay factor (default 0.6)
        - i = recency index (0 = most recent)
        """
        w_s = settings.SEMANTIC_WEIGHT
        w_e = settings.EPISODIC_WEIGHT
        alpha = settings.EPISODIC_DECAY
        
        merged = w_s * semantic
        
        if episodic_list:
            # Apply decay: most recent gets highest weight
            total_episodic = np.zeros_like(semantic)
            decay_sum = 0
            
            for i, e in enumerate(episodic_list):
                decay_factor = alpha ** i
                total_episodic += decay_factor * e
                decay_sum += decay_factor
            
            if decay_sum > 0:
                merged += w_e * (total_episodic / decay_sum)
        
        # Normalize
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        
        return merged
    
    async def delete_character(self, character_id: str):
        """
        Delete all vectors for a character (GDPR compliance).
        """
        semantic_id = f"sem_{character_id}"
        episodic_prefix = f"epi_{character_id}_"
        
        if self.use_pinecone:
            # Delete semantic
            self.index.delete(ids=[semantic_id], namespace="semantic")
            
            # Delete all episodic (Pinecone requires listing first)
            # This is a simplified version - production would need pagination
            self.index.delete(
                filter={"character_id": character_id},
                namespace="episodic"
            )
        else:
            self.local_store.delete(semantic_id)
            self.local_store.delete_by_prefix(episodic_prefix)
        
        print(f"[OK] Deleted all vectors for character: {character_id}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.use_pinecone:
            stats = self.index.describe_index_stats()
            return {
                "backend": "pinecone",
                "total_vectors": stats.total_vector_count,
                "namespaces": dict(stats.namespaces)
            }
        else:
            return {
                "backend": "local_faiss" if FAISS_AVAILABLE else "local_numpy",
                "semantic_count": len(self.local_store.semantic_vectors),
                "episodic_count": len(self.local_store.episodic_vectors),
                "total_vectors": len(self.local_store.semantic_vectors) + len(self.local_store.episodic_vectors)
            }
