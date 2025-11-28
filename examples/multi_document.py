#!/usr/bin/env python
"""Multi-document analysis example for SmartDoc Analyst.

This script demonstrates how to analyze multiple documents
and generate comparative reports.
"""

import asyncio
from src.core.system import SmartDocAnalyst


# Sample documents from different domains
MULTI_DOMAIN_DOCUMENTS = [
    {
        "content": """Artificial Intelligence in Healthcare: A Comprehensive Overview
        
AI is revolutionizing healthcare delivery across multiple domains. Machine learning 
algorithms are now capable of diagnosing diseases from medical images with accuracy 
matching or exceeding human specialists.

Key Applications:
1. Medical Imaging: AI systems analyze CT scans, MRIs, and X-rays to detect abnormalities
2. Drug Discovery: ML accelerates identification of potential drug candidates by 40%
3. Clinical Decision Support: AI assists physicians in treatment planning
4. Predictive Analytics: Models predict patient readmission and disease progression

The global AI in healthcare market is projected to reach $45.2 billion by 2026, 
growing at 44.9% CAGR. Recent advancements include GPT-4 passing medical licensing 
exams and FDA-approved AI diagnostic tools reaching over 500 as of 2024.""",
        "metadata": {
            "source": "ai_healthcare_report.pdf",
            "title": "AI in Healthcare Report 2024",
            "domain": "healthcare",
            "year": 2024
        }
    },
    {
        "content": """Q4 2024 Financial Market Analysis

Global financial markets experienced significant volatility in Q4 2024, driven by 
geopolitical tensions, central bank policy shifts, and evolving economic conditions.

Key Trends:
1. Interest Rates: Federal Reserve maintained rates at 5.25-5.5% with hints of cuts
2. Inflation: US CPI moderated to 3.1%, approaching but not reaching 2% target
3. AI Investment: Tech sector led by AI-related stocks with 45% average increase
4. Market Performance: S&P 500 +8.5%, NASDAQ +12.1%

The integration of AI technologies in financial services is accelerating, with 
major institutions deploying ML models for trading, risk assessment, and fraud 
detection. Investment in AI startups reached $50 billion in 2024.""",
        "metadata": {
            "source": "financial_analysis_q4.pdf",
            "title": "Q4 2024 Market Analysis",
            "domain": "finance",
            "year": 2024
        }
    },
    {
        "content": """Modern Software Architecture: Patterns and Best Practices

Software architecture continues to evolve with cloud-native and microservices 
patterns dominating enterprise development.

Core Patterns:
1. Microservices: Decompose applications into independently deployable services
2. Event-Driven: Use events for loose coupling between components
3. AI/ML Integration: Embedding AI capabilities into software systems
4. Domain-Driven Design: Align software with business domain models

Cloud-Native Principles:
- Containerization: Docker and Kubernetes for deployment consistency
- Service Mesh: Istio, Linkerd for inter-service communication
- Observability: Distributed tracing, logging, metrics (OpenTelemetry)
- AI Ops: Using ML for automated operations and monitoring""",
        "metadata": {
            "source": "software_architecture_guide.pdf",
            "title": "Software Architecture Guide 2024",
            "domain": "technology",
            "year": 2024
        }
    }
]


async def main():
    """Run multi-document analysis demonstration."""
    print("=" * 70)
    print("SmartDoc Analyst - Multi-Document Analysis Example")
    print("=" * 70)
    print()
    
    # Initialize the system
    print("Initializing SmartDoc Analyst...")
    analyst = SmartDocAnalyst()
    print("✓ System initialized\n")
    
    # Ingest all documents
    print("Ingesting documents from multiple domains...")
    result = analyst.ingest_documents(MULTI_DOMAIN_DOCUMENTS)
    print(f"✓ Added {result['added']} documents")
    for doc in MULTI_DOMAIN_DOCUMENTS:
        print(f"  - {doc['metadata']['title']} ({doc['metadata']['domain']})")
    print()
    
    # Cross-domain analysis queries
    queries = [
        "How is AI being adopted across different industries?",
        "What are the common technology trends mentioned across these documents?",
        "Compare the investment and growth patterns in AI across healthcare and finance",
    ]
    
    print("=" * 70)
    print("Running Cross-Domain Analyses")
    print("=" * 70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 60)
        
        result = await analyst.analyze(query, include_web_search=False)
        
        if result['success']:
            print(f"✓ Analysis completed in {result.get('execution_time_ms', 0):.2f}ms")
            print(f"✓ Processing stages: {', '.join(result.get('processing_stages', []))}")
            print(f"✓ Sources used: {len(result.get('sources', []))}")
            
            # Show quality score if available
            if result.get('quality_score'):
                print(f"✓ Quality score: {result['quality_score']}")
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print()
    print("=" * 70)
    
    # Search for specific topics
    print("\nSearching for specific topics across documents...")
    topics = ["AI adoption", "market growth", "technology trends"]
    
    for topic in topics:
        results = await analyst.search(topic, k=3)
        docs = results.get('documents', [])
        print(f"\n'{topic}': Found {len(docs)} matching documents")
        for doc in docs[:2]:
            title = doc.get('metadata', {}).get('title', 'Unknown')
            domain = doc.get('metadata', {}).get('domain', 'Unknown')
            print(f"  - {title} ({domain})")
    
    print()
    print("=" * 70)
    print("Multi-document analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
