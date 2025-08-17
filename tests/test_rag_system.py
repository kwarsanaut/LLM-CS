"""
Unit tests for RAG System components
"""

import unittest
import tempfile
import os
import json
import yaml
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag_system import (
    DocumentProcessor, 
    VectorStore, 
    IntentClassifier, 
    RAGSystem, 
    DocumentChunk,
    RetrievalResult
)

class TestDocumentProcessor(unittest.TestCase):
    """Test DocumentProcessor class"""
    
    def setUp(self):
        self.config = {
            'rag': {
                'retrieval': {
                    'chunk_size': 100,
                    'chunk_overlap': 20
                }
            }
        }
        self.processor = DocumentProcessor(self.config)
    
    def test_chunk_document(self):
        """Test document chunking"""
        content = """# KTP Requirements
        
## KTP BARU
Persyaratan untuk KTP baru:
1. KTP asli
2. Kartu Keluarga

## Prosedur
Langkah-langkah membuat KTP:
1. Datang ke kelurahan
2. Bawa dokumen lengkap
"""
        
        chunks = self.processor._chunk_document(content, "ktp", "ktp_requirements.txt")
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], DocumentChunk)
        self.assertEqual(chunks[0].document_type, "ktp")
        self.assertEqual(chunks[0].source_file, "ktp_requirements.txt")
    
    def test_split_text(self):
        """Test text splitting"""
        text = " ".join([f"word{i}" for i in range(50)])  # 50 words
        chunks = self.processor._split_text(text, chunk_size=20, overlap=5)
        
        self.assertGreater(len(chunks), 1)
        # Check overlap
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        self.assertGreater(len(first_chunk_words), 0)
        self.assertGreater(len(second_chunk_words), 0)

class TestIntentClassifier(unittest.TestCase):
    """Test IntentClassifier class"""
    
    def setUp(self):
        self.config = {}
        self.classifier = IntentClassifier(self.config)
    
    def test_classify_ktp_intent(self):
        """Test KTP intent classification"""
        query = "Saya mau buat KTP baru"
        intent = self.classifier.classify_intent(query)
        self.assertEqual(intent, "ktp")
    
    def test_classify_sim_intent(self):
        """Test SIM intent classification"""
        query = "Cara perpanjang SIM yang expired"
        intent = self.classifier.classify_intent(query)
        self.assertEqual(intent, "sim")
    
    def test_classify_passport_intent(self):
        """Test Passport intent classification"""
        query = "Mau bikin passport untuk ke luar negeri"
        intent = self.classifier.classify_intent(query)
        self.assertEqual(intent, "passport")
    
    def test_classify_general_intent(self):
        """Test general intent classification"""
        query = "Halo apa kabar"
        intent = self.classifier.classify_intent(query)
        self.assertEqual(intent, "general")

class TestVectorStore(unittest.TestCase):
    """Test VectorStore class"""
    
    def setUp(self):
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'rag': {
                'vector_db': {
                    'collection_name': 'test_collection',
                    'persist_directory': self.temp_dir
                },
                'embedding': {
                    'model_name': 'all-MiniLM-L6-v2'
                }
            }
        }
    
    def tearDown(self):
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.rag_system.SentenceTransformer')
    @patch('src.rag_system.chromadb')
    def test_vector_store_initialization(self, mock_chromadb, mock_sentence_transformer):
        """Test VectorStore initialization"""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        vector_store = VectorStore(self.config)
        
        self.assertIsNotNone(vector_store.embedding_model)
        mock_sentence_transformer.assert_called_once()
    
    @patch('src.rag_system.SentenceTransformer')
    @patch('src.rag_system.chromadb')
    def test_search_functionality(self, mock_chromadb, mock_sentence_transformer):
        """Test search functionality"""
        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]  # Mock embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['Test document content']],
            'metadatas': [[{'document_type': 'ktp', 'source_file': 'test.txt'}]],
            'distances': [[0.1]]
        }
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        vector_store = VectorStore(self.config)
        results = vector_store.search("test query", top_k=1)
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].document_type, 'ktp')

class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for RAG System"""
    
    def setUp(self):
        # Create temporary files and directories
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config = {
            'model': {'name': 'test-model'},
            'tokenizer': {'max_length': 512},
            'rag': {
                'vector_db': {
                    'collection_name': 'test_collection',
                    'persist_directory': os.path.join(self.temp_dir, 'vector_db')
                },
                'embedding': {
                    'model_name': 'all-MiniLM-L6-v2'
                },
                'retrieval': {
                    'top_k': 3,
                    'similarity_threshold': 0.7,
                    'chunk_size': 512,
                    'chunk_overlap': 50
                }
            },
            'data': {
                'knowledge_base_dir': os.path.join(self.temp_dir, 'kb')
            },
            'paths': {
                'model_output_dir': os.path.join(self.temp_dir, 'models')
            },
            'generation': {
                'max_new_tokens': 256,
                'temperature': 0.7,
                'top_p': 0.9,
                'do_sample': True,
                'repetition_penalty': 1.1
            }
        }
        
        # Create config file
        self.config_file = os.path.join(self.temp_dir, 'config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create knowledge base directory and files
        kb_dir = self.config['data']['knowledge_base_dir']
        os.makedirs(kb_dir, exist_ok=True)
        
        # Create minimal knowledge base files
        kb_content = """# Test Knowledge Base
## Requirements
1. Document A
2. Document B
"""
        
        for doc_type in ['ktp', 'sim', 'passport', 'akta_kelahiran']:
            with open(os.path.join(kb_dir, f'{doc_type}_requirements.txt'), 'w') as f:
                f.write(kb_content)
    
    def tearDown(self):
        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.rag_system.AutoTokenizer')
    @patch('src.rag_system.AutoModelForCausalLM')
    @patch('src.rag_system.SentenceTransformer')
    @patch('src.rag_system.chromadb')
    def test_rag_system_initialization(self, mock_chromadb, mock_sentence_transformer, 
                                     mock_model, mock_tokenizer):
        """Test RAG system initialization"""
        # Mock all external dependencies
        mock_sentence_transformer.return_value = Mock()
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # This should not raise an exception
        rag_system = RAGSystem(self.config_file)
        
        self.assertIsNotNone(rag_system.document_processor)
        self.assertIsNotNone(rag_system.vector_store)
        self.assertIsNotNone(rag_system.intent_classifier)

class TestDataValidation(unittest.TestCase):
    """Test data validation utilities"""
    
    def test_conversation_data_validation(self):
        """Test conversation data format validation"""
        valid_conversation = {
            "id": "test_001",
            "document_type": "ktp", 
            "intent": "persyaratan_ktp_baru",
            "conversation": [
                {
                    "role": "user",
                    "message": "Test question"
                },
                {
                    "role": "assistant",
                    "message": "Test response"
                }
            ]
        }
        
        # Test required fields
        required_fields = ["id", "document_type", "intent", "conversation"]
        for field in required_fields:
            self.assertIn(field, valid_conversation)
        
        # Test conversation structure
        conversation = valid_conversation["conversation"]
        self.assertIsInstance(conversation, list)
        self.assertGreater(len(conversation), 0)
        
        for message in conversation:
            self.assertIn("role", message)
            self.assertIn("message", message)
            self.assertIn(message["role"], ["user", "assistant"])
    
    def test_invalid_conversation_data(self):
        """Test invalid conversation data detection"""
        invalid_conversations = [
            {},  # Empty dict
            {"id": "test"},  # Missing fields
            {
                "id": "test",
                "document_type": "ktp",
                "intent": "test",
                "conversation": []  # Empty conversation
            }
        ]
        
        for invalid_conv in invalid_conversations:
            required_fields = ["id", "document_type", "intent", "conversation"]
            missing_fields = [field for field in required_fields if field not in invalid_conv]
            
            if missing_fields:
                self.assertGreater(len(missing_fields), 0)

class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_valid_config_structure(self):
        """Test valid configuration structure"""
        config = {
            'model': {'name': 'test-model'},
            'training': {'num_epochs': 3},
            'rag': {
                'vector_db': {'collection_name': 'test'},
                'retrieval': {'top_k': 3}
            },
            'data': {'train_file': 'test.json'}
        }
        
        required_sections = ['model', 'rag', 'data']
        for section in required_sections:
            self.assertIn(section, config)
    
    def test_config_loading(self):
        """Test config loading from YAML"""
        config_data = {
            'model': {'name': 'test-model'},
            'rag': {'vector_db': {'collection_name': 'test'}},
            'data': {'train_file': 'test.json'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertEqual(loaded_config['model']['name'], 'test-model')
            self.assertEqual(loaded_config['rag']['vector_db']['collection_name'], 'test')
        finally:
            os.unlink(config_file)

if __name__ == '__main__':
    unittest.main()
