#!/usr/bin/env python
"""Basic usage example for SmartDoc Analyst.

This script demonstrates the core functionality of SmartDoc Analyst,
including document ingestion, analysis, and querying.
"""

import asyncio
import os
from src.core.system import SmartDocAnalyst
from src.agents.base_agent import AgentContext


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "content": """Artificial Intelligence in Healthcare: A Comprehensive Overview
        
AI is revolutionizing healthcare delivery across multiple domains. Machine learning 
algorithms are now capable of diagnosing diseases from medical images with accuracy 
matching or exceeding human specialists.

Key Applications:
1. Medical Imaging: AI systems analyze CT scans, MRIs, and X-rays
2. Drug Discovery: ML accelerates identification of potential drug candidates
3. Clinical Decision Support: AI assists physicians in treatment planning
4. Predictive Analytics: Models predict patient outcomes

The global AI in healthcare market is projected to reach $45.2 billion by 2026.""",
        "metadata": {
            "source": "ai_healthcare_report.pdf",
            "title": "AI in Healthcare Report 2024",
            "author": "Healthcare AI Institute"
        }
    },
    {
        "content": """Climate Change Policy Analysis: Global Perspectives

Climate change represents one of the most pressing challenges of our time.
International efforts centered around the Paris Agreement aim to limit global
warming to 1.5°C above pre-industrial levels.

Key Policy Mechanisms:
1. Carbon Pricing: 46 countries have implemented carbon taxes
2. Renewable Energy Mandates: Over 170 countries have targets
3. Green Finance: Climate-aligned investments reached $1.3 trillion
4. International Cooperation: Climate funds support developing nations""",
        "metadata": {
            "source": "climate_policy.pdf",
            "title": "Global Climate Policy Analysis",
            "author": "Environmental Policy Institute"
        }
    }
]


async def main():
    """Run basic usage demonstration."""
    print("=" * 60)
    print("SmartDoc Analyst - Basic Usage Example")
    print("=" * 60)
    print()
    
    # Initialize the system
    # Note: Set SMARTDOC_GEMINI_API_KEY environment variable for full functionality
    print("1. Initializing SmartDoc Analyst...")
    analyst = SmartDocAnalyst()
    print("   ✓ System initialized")
    print()
    
    # Ingest documents
    print("2. Ingesting sample documents...")
    result = analyst.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"   ✓ Added {result['added']} documents")
    print(f"   ✓ Document IDs: {result['document_ids']}")
    print()
    
    # Search documents
    print("3. Searching documents...")
    search_results = await analyst.search("AI healthcare applications", k=3)
    print(f"   ✓ Found {len(search_results.get('documents', []))} matching documents")
    for i, doc in enumerate(search_results.get('documents', [])[:2], 1):
        title = doc.get('metadata', {}).get('title', 'Unknown')
        print(f"   {i}. {title}")
    print()
    
    # Analyze a query
    print("4. Analyzing query: 'What are the key trends in AI healthcare?'")
    analysis = await analyst.analyze(
        "What are the key trends in AI healthcare?",
        include_web_search=False
    )
    print(f"   ✓ Success: {analysis['success']}")
    if analysis['success']:
        print(f"   ✓ Processing stages: {analysis.get('processing_stages', [])}")
        print(f"   ✓ Task ID: {analysis.get('task_id', 'N/A')}")
        print(f"   ✓ Execution time: {analysis.get('execution_time_ms', 0):.2f}ms")
    print()
    
    # Summarize text
    print("5. Summarizing text...")
    summary = await analyst.summarize(
        SAMPLE_DOCUMENTS[0]["content"],
        max_length=50
    )
    print(f"   ✓ Summary: {summary[:100]}...")
    print()
    
    # Get system stats
    print("6. System Statistics:")
    stats = analyst.get_stats()
    print(f"   ✓ Memory stats: {stats['memory']['total_entries']} entries")
    print(f"   ✓ Metrics collected: {len(stats.get('metrics', {}))} types")
    print()
    
    print("=" * 60)
    print("Basic usage demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
