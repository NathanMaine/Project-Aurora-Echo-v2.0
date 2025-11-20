#!/usr/bin/env python3
"""Test script to verify Ollama integration with Aurora Echo."""

import asyncio
import os
from dotenv import load_dotenv
from services.llm_service import LLMService
import httpx

# Load environment variables
load_dotenv()

async def test_ollama_direct():
    """Test Ollama API directly."""
    print("🔗 Testing Ollama API directly...")
    try:
        # Test with the same httpx configuration as VLLMProvider
        async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=30.0) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "Hello, test message"}]
            })
            if response.status_code == 200:
                print("✅ Direct API call successful!")
                return True
            else:
                print(f"❌ API call failed with status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Direct API call failed: {e}")
        return False

async def test_ollama_integration():
    """Test that Aurora Echo can communicate with Ollama."""
    print("🧪 Testing Aurora Echo + Ollama Integration")
    print("=" * 50)

    # First test direct API connection
    if not await test_ollama_direct():
        print("❌ Cannot connect to Ollama API directly")
        return False

    # Initialize LLM service
    llm_service = LLMService()
    print("✅ LLM Service initialized")

    # Test a simple summary
    test_transcript = """
    Meeting transcript:
    John: Good morning everyone. Let's discuss the quarterly results.
    Sarah: The Q3 numbers look strong, with 15% growth in revenue.
    Mike: I think we should focus on expanding into the European market.
    John: Agreed. Let's schedule a follow-up meeting for next week.
    """

    print("📝 Testing LLM summarization with Aurora Echo...")
    try:
        result = await llm_service.summarize_meeting(test_transcript)
        if result:
            print("✅ LLM call successful!")
            print(f"📄 Summary: {result.get('content', '')[:200]}...")
            return True
        else:
            print("❌ LLM call returned None")
            return False
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama_integration())
    if success:
        print("\n🎉 Aurora Echo + Ollama integration test PASSED!")
        print("Your local LLM setup is working correctly.")
    else:
        print("\n❌ Integration test FAILED!")
        print("Check your Ollama setup and Aurora Echo configuration.")