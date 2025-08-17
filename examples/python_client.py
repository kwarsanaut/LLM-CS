"""
Python client examples for IndoBERT Document Customer Service API
"""

import requests
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class ChatResponse:
    """Response from chat API"""
    response: str
    intent: str
    conversation_id: str
    confidence: Optional[float] = None
    timestamp: str = ""

class IndoBERTDocumentClient:
    """Python client for IndoBERT Document CS API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'IndoBERT-Client/1.0'
        })
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def chat(self, query: str, conversation_id: Optional[str] = None, 
             include_context: bool = False) -> ChatResponse:
        """Send chat message"""
        payload = {
            "query": query,
            "include_context": include_context
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return ChatResponse(
            response=data["response"],
            intent=data["intent"],
            conversation_id=data["conversation_id"],
            confidence=data.get("confidence"),
            timestamp=data["timestamp"]
        )
    
    def get_conversation(self, conversation_id: str) -> Dict:
        """Get conversation history"""
        response = self.session.get(f"{self.base_url}/conversations/{conversation_id}")
        response.raise_for_status()
        return response.json()
    
    def get_intents(self) -> Dict:
        """Get supported intents"""
        response = self.session.get(f"{self.base_url}/intents")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get API statistics"""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def run_evaluation(self, test_file: str = "data/raw/conversations_test.json", 
                      save_results: bool = True) -> str:
        """Run model evaluation"""
        payload = {
            "test_file": test_file,
            "save_results": save_results
        }
        
        response = self.session.post(f"{self.base_url}/evaluate", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["evaluation_id"]
    
    def get_evaluation_status(self, evaluation_id: str) -> Dict:
        """Get evaluation status"""
        response = self.session.get(f"{self.base_url}/evaluation/{evaluation_id}")
        response.raise_for_status()
        return response.json()

class AsyncIndoBERTClient:
    """Async Python client for better performance"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    async def chat(self, query: str, conversation_id: Optional[str] = None) -> ChatResponse:
        """Async chat method"""
        payload = {
            "query": query,
            "include_context": False
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/chat", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                
                return ChatResponse(
                    response=data["response"],
                    intent=data["intent"], 
                    conversation_id=data["conversation_id"],
                    confidence=data.get("confidence"),
                    timestamp=data["timestamp"]
                )
    
    async def batch_chat(self, queries: List[str]) -> List[ChatResponse]:
        """Process multiple queries concurrently"""
        tasks = [self.chat(query) for query in queries]
        return await asyncio.gather(*tasks)

def example_basic_usage():
    """Basic usage example"""
    print("🔥 Basic Usage Example")
    print("=" * 50)
    
    # Initialize client
    client = IndoBERTDocumentClient()
    
    # Health check
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API not available: {e}")
        return
    
    # Example queries
    queries = [
        "Syarat bikin KTP baru apa aja ya?",
        "SIM saya mau expired, cara perpanjangnya gimana?",
        "Anak saya mau ke luar negeri, perlu passport ga?",
        "Akta kelahiran hilang bisa diganti ga?"
    ]
    
    conversation_id = None
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            response = client.chat(query, conversation_id)
            conversation_id = response.conversation_id  # Continue conversation
            
            print(f"🤖 Intent: {response.intent}")
            print(f"📋 Response: {response.response[:200]}...")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")

def example_conversation_flow():
    """Example of maintaining conversation context"""
    print("\n💬 Conversation Flow Example")
    print("=" * 50)
    
    client = IndoBERTDocumentClient()
    
    # Start conversation
    response1 = client.chat("Mau bikin KTP baru nih")
    conversation_id = response1.conversation_id
    
    print(f"User: Mau bikin KTP baru nih")
    print(f"Bot: {response1.response}")
    
    # Follow-up questions
    follow_ups = [
        "Kalau KTP-nya hilang gimana?",
        "Biayanya berapa?",
        "Butuh waktu berapa lama?"
    ]
    
    for follow_up in follow_ups:
        response = client.chat(follow_up, conversation_id)
        print(f"\nUser: {follow_up}")
        print(f"Bot: {response.response}")
    
    # Get full conversation history
    history = client.get_conversation(conversation_id)
    print(f"\n📚 Total messages in conversation: {len(history['messages'])}")

async def example_async_usage():
    """Async usage example for better performance"""
    print("\n⚡ Async Usage Example")
    print("=" * 50)
    
    client = AsyncIndoBERTClient()
    
    # Batch processing
    queries = [
        "Syarat SIM A baru apa aja?",
        "Passport anak umur 5 tahun gimana?", 
        "KTP pindah domisili prosedurnya?",
        "Akta kelahiran terlambat denda ga?"
    ]
    
    start_time = time.time()
    responses = await client.batch_chat(queries)
    end_time = time.time()
    
    print(f"⏱️  Processed {len(queries)} queries in {end_time - start_time:.2f} seconds")
    
    for query, response in zip(queries, responses):
        print(f"\n📝 {query}")
        print(f"🎯 Intent: {response.intent}")
        print(f"📋 Response: {response.response[:150]}...")

def example_evaluation_workflow():
    """Example of running evaluation"""
    print("\n📊 Evaluation Workflow Example")
    print("=" * 50)
    
    client = IndoBERTDocumentClient()
    
    # Start evaluation
    try:
        eval_id = client.run_evaluation()
        print(f"🔄 Started evaluation: {eval_id}")
        
        # Check status
        import time
        while True:
            status = client.get_evaluation_status(eval_id)
            print(f"📈 Status: {status['status']}")
            
            if status['status'] == 'completed':
                results = status['results']
                print(f"✅ Overall Score: {results['overall_score']:.3f}")
                print(f"🎯 Intent Accuracy: {results['intent_accuracy']:.3f}")
                print(f"📝 BLEU Score: {results['bleu_score']:.3f}")
                break
            elif status['status'] == 'failed':
                print("❌ Evaluation failed")
                break
            
            time.sleep(5)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Evaluation error: {e}")

def example_error_handling():
    """Example of proper error handling"""
    print("\n🛡️  Error Handling Example")
    print("=" * 50)
    
    client = IndoBERTDocumentClient("http://invalid-url:9999")
    
    try:
        response = client.chat("Test query")
        print(f"Response: {response.response}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: API server not reachable")
        
    except requests.exceptions.Timeout:
        print("❌ Timeout error: Request took too long")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e.response.status_code}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ General request error: {e}")

def example_custom_configuration():
    """Example with custom configuration"""
    print("\n⚙️  Custom Configuration Example")
    print("=" * 50)
    
    # Custom client with timeout and retries
    session = requests.Session()
    session.timeout = 30
    
    # Add retry logic
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    client = IndoBERTDocumentClient()
    client.session = session
    
    try:
        response = client.chat("Test dengan retry mechanism")
        print(f"✅ Success: {response.intent}")
    except Exception as e:
        print(f"❌ Failed after retries: {e}")

class DocumentAssistantBot:
    """High-level bot interface"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.client = IndoBERTDocumentClient(api_url)
        self.conversations = {}
    
    def ask(self, user_id: str, question: str) -> str:
        """Simple ask interface"""
        conversation_id = self.conversations.get(user_id)
        
        try:
            response = self.client.chat(question, conversation_id)
            self.conversations[user_id] = response.conversation_id
            return response.response
            
        except Exception as e:
            return f"Maaf, terjadi kesalahan: {str(e)}"
    
    def reset_conversation(self, user_id: str):
        """Reset user conversation"""
        if user_id in self.conversations:
            del self.conversations[user_id]

def example_bot_usage():
    """Example bot usage"""
    print("\n🤖 Bot Interface Example")
    print("=" * 50)
    
    bot = DocumentAssistantBot()
    
    # Simulate multiple users
    users = ["user1", "user2"]
    
    for user in users:
        print(f"\n👤 {user}:")
        
        questions = [
            "Halo, mau tanya tentang KTP",
            "Syaratnya apa aja?",
            "Kalau belum punya akta kelahiran gimana?"
        ]
        
        for question in questions:
            answer = bot.ask(user, question)
            print(f"  Q: {question}")
            print(f"  A: {answer[:100]}...")

if __name__ == "__main__":
    print("🏛️  IndoBERT Document CS - Python Client Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_conversation_flow()
    
    # Async example (requires asyncio)
    asyncio.run(example_async_usage())
    
    example_evaluation_workflow()
    example_error_handling()
    example_custom_configuration()
    example_bot_usage()
    
    print("\n🎉 All examples completed!")
    print("\n📚 More examples:")
    print("- JavaScript: examples/javascript_client.js")
    print("- React: examples/react_component.jsx")
    print("- cURL: examples/curl_examples.sh")

"""
JavaScript Client Example (examples/javascript_client.js):

class IndoBERTClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async chat(query, conversationId = null) {
        const payload = {
            query: query,
            include_context: false
        };
        
        if (conversationId) {
            payload.conversation_id = conversationId;
        }
        
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async health() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
}

// Usage
const client = new IndoBERTClient();

client.chat('Syarat bikin KTP baru apa aja ya?')
    .then(response => {
        console.log('Intent:', response.intent);
        console.log('Response:', response.response);
    })
    .catch(error => {
        console.error('Error:', error);
    });
"""
