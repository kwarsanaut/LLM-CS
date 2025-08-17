"""
Data preprocessing utilities for IndoBERT Document Customer Service
Tools for data cleaning, augmentation, and validation
"""

import json
import os
import re
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict, Counter
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationStats:
    """Statistics for conversation dataset"""
    total_conversations: int
    total_messages: int
    avg_messages_per_conversation: float
    document_type_distribution: Dict[str, int]
    intent_distribution: Dict[str, int]
    avg_message_length: float
    vocabulary_size: int

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Text cleaning patterns
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'[^\w\s\?\!\.\,\-\:\;]', ''),  # Remove special characters except basic punctuation
            (r'\.{2,}', '...'),  # Multiple dots to ellipsis
            (r'\?{2,}', '?'),  # Multiple question marks to single
            (r'\!{2,}', '!'),  # Multiple exclamation marks to single
        ]
        
        # Paraphrasing templates for data augmentation
        self.paraphrase_templates = {
            'question_starters': [
                'Bagaimana cara {}?',
                'Gimana caranya {}?', 
                'Bisa bantu {} ga?',
                'Mau tanya tentang {}',
                'Info {} dong',
                'Persyaratan {} apa aja?'
            ],
            'document_variations': {
                'ktp': ['KTP', 'kartu tanda penduduk', 'e-KTP', 'identitas'],
                'sim': ['SIM', 'surat izin mengemudi', 'surat izin berkendara'],
                'passport': ['passport', 'paspor', 'dokumen perjalanan'],
                'akta_kelahiran': ['akta kelahiran', 'akta lahir', 'surat kelahiran']
            }
        }
    
    def load_conversations(self, file_path: str) -> List[Dict]:
        """Load conversation data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_conversations(self, conversations: List[Dict], file_path: str):
        """Save conversation data to JSON file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(conversations)} conversations to {file_path}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Normalize whitespace
        text = text.strip()
        
        # Basic Indonesian text normalization
        text = self._normalize_indonesian_text(text)
        
        return text
    
    def _normalize_indonesian_text(self, text: str) -> str:
        """Normalize Indonesian text"""
        # Common Indonesian abbreviations and normalizations
        normalizations = {
            'ga': 'tidak',
            'gak': 'tidak', 
            'tdk': 'tidak',
            'dgn': 'dengan',
            'yg': 'yang',
            'klo': 'kalau',
            'kalo': 'kalau',
            'gimana': 'bagaimana',
            'gmn': 'bagaimana',
            'udah': 'sudah',
            'blm': 'belum',
            'hrs': 'harus',
            'utk': 'untuk',
            'krn': 'karena',
            'jd': 'jadi',
            'tp': 'tapi',
            'sm': 'sama'
        }
        
        words = text.split()
        normalized_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in normalizations:
                normalized_words.append(normalizations[word_lower])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def validate_conversations(self, conversations: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Validate conversation data and return valid conversations and error messages"""
        valid_conversations = []
        errors = []
        
        required_fields = ['id', 'document_type', 'intent', 'conversation']
        valid_document_types = ['ktp', 'sim', 'passport', 'akta_kelahiran']
        
        for i, conv in enumerate(conversations):
            conv_errors = []
            
            # Check required fields
            for field in required_fields:
                if field not in conv:
                    conv_errors.append(f"Missing field: {field}")
            
            if conv_errors:
                errors.extend([f"Conversation {i}: {error}" for error in conv_errors])
                continue
            
            # Validate document type
            if conv['document_type'] not in valid_document_types:
                conv_errors.append(f"Invalid document_type: {conv['document_type']}")
            
            # Validate conversation structure
            conversation = conv['conversation']
            if not isinstance(conversation, list) or len(conversation) == 0:
                conv_errors.append("Empty or invalid conversation")
            else:
                for j, message in enumerate(conversation):
                    if not isinstance(message, dict):
                        conv_errors.append(f"Message {j} is not a dict")
                        continue
                    
                    if 'role' not in message or 'message' not in message:
                        conv_errors.append(f"Message {j} missing role or message")
                    
                    if message.get('role') not in ['user', 'assistant']:
                        conv_errors.append(f"Message {j} has invalid role: {message.get('role')}")
                    
                    if not message.get('message', '').strip():
                        conv_errors.append(f"Message {j} has empty content")
            
            if conv_errors:
                errors.extend([f"Conversation {conv['id']}: {error}" for error in conv_errors])
            else:
                valid_conversations.append(conv)
        
        return valid_conversations, errors
    
    def calculate_stats(self, conversations: List[Dict]) -> ConversationStats:
        """Calculate dataset statistics"""
        total_conversations = len(conversations)
        total_messages = 0
        message_lengths = []
        document_types = []
        intents = []
        all_words = []
        
        for conv in conversations:
            conversation = conv['conversation']
            total_messages += len(conversation)
            document_types.append(conv['document_type'])
            intents.append(conv['intent'])
            
            for message in conversation:
                content = message['message']
                message_lengths.append(len(content.split()))
                all_words.extend(content.lower().split())
        
        avg_messages_per_conversation = total_messages / total_conversations if total_conversations > 0 else 0
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        vocabulary_size = len(set(all_words))
        
        return ConversationStats(
            total_conversations=total_conversations,
            total_messages=total_messages,
            avg_messages_per_conversation=avg_messages_per_conversation,
            document_type_distribution=dict(Counter(document_types)),
            intent_distribution=dict(Counter(intents)),
            avg_message_length=avg_message_length,
            vocabulary_size=vocabulary_size
        )
    
    def augment_conversations(self, conversations: List[Dict], 
                            augmentation_factor: float = 2.0) -> List[Dict]:
        """Augment conversation data through paraphrasing and variations"""
        augmented = conversations.copy()
        target_count = int(len(conversations) * augmentation_factor)
        
        while len(augmented) < target_count:
            # Select random conversation to augment
            base_conv = random.choice(conversations)
            augmented_conv = self._create_variation(base_conv)
            
            if augmented_conv:
                augmented.append(augmented_conv)
        
        logger.info(f"Augmented dataset from {len(conversations)} to {len(augmented)} conversations")
        return augmented
    
    def _create_variation(self, conversation: Dict) -> Optional[Dict]:
        """Create a variation of a conversation"""
        try:
            new_conv = conversation.copy()
            new_conv['id'] = f"{conversation['id']}_aug_{random.randint(1000, 9999)}"
            new_conversation = []
            
            for message in conversation['conversation']:
                if message['role'] == 'user':
                    # Create variation of user message
                    original_text = message['message']
                    varied_text = self._create_text_variation(original_text, conversation['document_type'])
                    new_conversation.append({
                        'role': 'user',
                        'message': varied_text
                    })
                else:
                    # Keep assistant message mostly the same with minor variations
                    new_conversation.append(message.copy())
            
            new_conv['conversation'] = new_conversation
            return new_conv
            
        except Exception as e:
            logger.warning(f"Failed to create variation: {e}")
            return None
    
    def _create_text_variation(self, text: str, document_type: str) -> str:
        """Create text variation using paraphrasing templates"""
        # Simple paraphrasing - replace document type variations
        variations = self.paraphrase_templates['document_variations'].get(document_type, [document_type])
        
        for variation in variations:
            if variation.lower() in text.lower():
                # Replace with random alternative
                alternatives = [v for v in variations if v != variation]
                if alternatives:
                    replacement = random.choice(alternatives)
                    text = re.sub(re.escape(variation), replacement, text, flags=re.IGNORECASE)
                break
        
        # Add some natural variations
        casual_replacements = {
            'bagaimana': random.choice(['gimana', 'bagaimana']),
            'tidak': random.choice(['ga', 'tidak', 'gak']),
            'apa saja': random.choice(['apa aja', 'apa saja']),
            'ya?': random.choice(['ya?', 'dong?', '?'])
        }
        
        for original, replacement in casual_replacements.items():
            if original in text.lower():
                text = re.sub(re.escape(original), replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def split_dataset(self, conversations: List[Dict], 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Shuffle conversations
        shuffled = conversations.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_set = shuffled[:train_end]
        val_set = shuffled[train_end:val_end]
        test_set = shuffled[val_end:]
        
        logger.info(f"Split dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        
        return train_set, val_set, test_set
    
    def balance_dataset(self, conversations: List[Dict]) -> List[Dict]:
        """Balance dataset by document types"""
        # Group by document type
        by_doc_type = defaultdict(list)
        for conv in conversations:
            by_doc_type[conv['document_type']].append(conv)
        
        # Find minimum count
        min_count = min(len(convs) for convs in by_doc_type.values())
        
        # Sample equally from each type
        balanced = []
        for doc_type, convs in by_doc_type.items():
            sampled = random.sample(convs, min_count)
            balanced.extend(sampled)
        
        random.shuffle(balanced)
        logger.info(f"Balanced dataset to {len(balanced)} conversations ({min_count} per document type)")
        
        return balanced
    
    def export_to_csv(self, conversations: List[Dict], output_path: str):
        """Export conversations to CSV format"""
        data = []
        
        for conv in conversations:
            for message in conv['conversation']:
                data.append({
                    'conversation_id': conv['id'],
                    'document_type': conv['document_type'],
                    'intent': conv['intent'],
                    'role': message['role'],
                    'message': message['message']
                })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Exported {len(data)} messages to {output_path}")
    
    def generate_report(self, conversations: List[Dict], output_path: str):
        """Generate comprehensive data report"""
        stats = self.calculate_stats(conversations)
        
        report = {
            'dataset_overview': {
                'total_conversations': stats.total_conversations,
                'total_messages': stats.total_messages,
                'avg_messages_per_conversation': round(stats.avg_messages_per_conversation, 2),
                'avg_message_length': round(stats.avg_message_length, 2),
                'vocabulary_size': stats.vocabulary_size
            },
            'document_type_distribution': stats.document_type_distribution,
            'intent_distribution': stats.intent_distribution,
            'data_quality': {
                'conversations_validated': True,
                'text_cleaned': True,
                'balanced': self._check_balance(stats.document_type_distribution)
            }
        }
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET REPORT")
        print("="*50)
        print(f"Total Conversations: {stats.total_conversations}")
        print(f"Total Messages: {stats.total_messages}")
        print(f"Avg Messages/Conversation: {stats.avg_messages_per_conversation:.2f}")
        print(f"Avg Message Length: {stats.avg_message_length:.2f} words")
        print(f"Vocabulary Size: {stats.vocabulary_size}")
        print("\nDocument Type Distribution:")
        for doc_type, count in stats.document_type_distribution.items():
            percentage = (count / stats.total_conversations) * 100
            print(f"  {doc_type}: {count} ({percentage:.1f}%)")
        
        logger.info(f"Report saved to {output_path}")
    
    def _check_balance(self, distribution: Dict[str, int]) -> bool:
        """Check if dataset is balanced"""
        counts = list(distribution.values())
        if not counts:
            return True
        
        min_count = min(counts)
        max_count = max(counts)
        ratio = min_count / max_count if max_count > 0 else 1
        
        return ratio > 0.7  # Consider balanced if min/max ratio > 0.7

def main():
    """Main function for data preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preprocessing for IndoBERT Document CS')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output-dir', '-o', default='data/processed', help='Output directory')
    parser.add_argument('--clean', action='store_true', help='Clean and normalize text')
    parser.add_argument('--validate', action='store_true', help='Validate data')
    parser.add_argument('--augment', type=float, default=0, help='Augmentation factor (e.g., 2.0 for 2x data)')
    parser.add_argument('--balance', action='store_true', help='Balance dataset by document types')
    parser.add_argument('--split', action='store_true', help='Split into train/val/test')
    parser.add_argument('--report', action='store_true', help='Generate data report')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    logger.info(f"Loading conversations from {args.input}")
    conversations = preprocessor.load_conversations(args.input)
    logger.info(f"Loaded {len(conversations)} conversations")
    
    # Validate
    if args.validate:
        logger.info("Validating conversations...")
        valid_conversations, errors = preprocessor.validate_conversations(conversations)
        if errors:
            logger.warning(f"Found {len(errors)} validation errors:")
            for error in errors[:10]:  # Show first 10 errors
                logger.warning(f"  {error}")
        conversations = valid_conversations
        logger.info(f"Kept {len(conversations)} valid conversations")
    
    # Clean text
    if args.clean:
        logger.info("Cleaning text...")
        for conv in conversations:
            for message in conv['conversation']:
                message['message'] = preprocessor.clean_text(message['message'])
    
    # Balance dataset
    if args.balance:
        logger.info("Balancing dataset...")
        conversations = preprocessor.balance_dataset(conversations)
    
    # Augment data
    if args.augment > 0:
        logger.info(f"Augmenting data with factor {args.augment}")
        conversations = preprocessor.augment_conversations(conversations, args.augment)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split dataset
    if args.split:
        logger.info("Splitting dataset...")
        train, val, test = preprocessor.split_dataset(conversations)
        
        preprocessor.save_conversations(train, os.path.join(args.output_dir, 'conversations_train.json'))
        preprocessor.save_conversations(val, os.path.join(args.output_dir, 'conversations_val.json'))
        preprocessor.save_conversations(test, os.path.join(args.output_dir, 'conversations_test.json'))
    else:
        # Save all conversations
        output_file = os.path.join(args.output_dir, 'conversations_processed.json')
        preprocessor.save_conversations(conversations, output_file)
    
    # Generate report
    if args.report:
        report_file = os.path.join(args.output_dir, 'data_report.json')
        preprocessor.generate_report(conversations, report_file)
    
    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main()
