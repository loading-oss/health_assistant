"""
记忆管理模块
包含记忆条目定义、存储与操作等功能
"""
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIM = EMBEDDING_MODEL.get_sentence_embedding_dimension()

class MemoryEntry:
    """记忆条目基础类"""
    def __init__(self, content: str, timestamp: str, memory_type: str, metadata: Dict = None):
        self.content = content
        self.timestamp = timestamp
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.embedding = EMBEDDING_MODEL

    def to_dict(self):
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "metadata": self.metadata
        }
    
    def __str__(self):
        return f"[{self.memory_type}] {self.timestamp}: {self.content}"

class BaseMemory:
    """记忆存储与操作基础类"""
    def __init__(self, index_path: str = "./data/memory_index.faiss", metadata_path: str = "./data/memory_metadata.pkl"):
        self.embedding = EMBEDDING_MODEL
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.metadata: List[MemoryEntry] = []

        self.index_path = index_path
        self.metadata_path = metadata_path

    def get_all_by_type(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        return self.metadata

    def add_entry(self, entry: MemoryEntry):
        """添加记忆条目"""
        # 生成嵌入向量
        embedding = self.embedding_model.encode([entry.content])[0].astype('float32')
        entry.embedding = embedding

        # 添加到 FAISS
        self.index.add(np.array([embedding]))

        # 添加到元数据
        self.metadata.append(entry)

    def delete_entry(self, index: int):
        """删除指定索引的记忆（谨慎操作）"""
        if 0 <= index < len(self.metadata):
            del self.metadata[index]
            self._rebuild_index()

    def update_entry(self, index: int, new_content: str):
        """更新记忆内容"""
        if 0 <= index < len(self.metadata):
            old_entry = self.metadata[index]
            new_entry = MemoryEntry(
                content=new_content,
                timestamp=old_entry.timestamp,
                memory_type=old_entry.memory_type,
                metadata=old_entry.metadata
            )
            self.metadata[index] = new_entry
            self._rebuild_index()

    def search(self, query: str, k: int = 5, filter_type: str = None) -> List[Tuple[MemoryEntry, float]]:
        """语义搜索最相似的记忆"""
        query_vec = self.embedding_model.encode([query])[0].astype('float32')
        D, I = self.index.search(np.array([query_vec]), k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx != -1 and idx < len(self.metadata):
                entry = self.metadata[idx]
                if filter_type is None or entry.memory_type == filter_type:
                    results.append((entry, dist))

        return results

    def save(self):
        """持久化记忆"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("💾 Memory saved.")

    def load(self):
        """加载记忆"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print("📁 Memory loaded.")
        else:
            print("🆕 No existing memory found, starting fresh.")

    def _rebuild_index(self):
        """重建 FAISS 索引（性能代价高，适用于小数据量）"""
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        vectors = []
        for entry in self.metadata:
            vec = self.embedding_model.encode([entry.content])[0].astype('float32')
            vectors.append(vec)
        if vectors:
            self.index.add(np.array(vectors))