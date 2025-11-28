"""Test cases for SmartDoc Analyst evaluation.

This module provides 20+ test cases covering various
scenarios for comprehensive system evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TestCase:
    """A single test case for evaluation.
    
    Attributes:
        name: Unique test case identifier.
        description: Human-readable description.
        difficulty: Difficulty level (easy, medium, hard).
        category: Test category.
        query: Test query string.
        documents: Sample documents for the test.
        expected_topics: Topics that should appear in response.
        expected_citations: Minimum citations expected.
        timeout_seconds: Maximum execution time.
        metadata: Additional test metadata.
    """
    name: str
    description: str
    difficulty: str
    category: str
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    expected_topics: List[str] = field(default_factory=list)
    expected_citations: int = 0
    timeout_seconds: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


# Sample documents for testing
SAMPLE_DOCUMENTS = {
    "ai_healthcare": {
        "content": """Artificial Intelligence in Healthcare: A Comprehensive Overview

AI is revolutionizing healthcare delivery across multiple domains. Machine learning algorithms 
are now capable of diagnosing diseases from medical images with accuracy matching or exceeding 
human specialists. Deep learning models have shown remarkable success in detecting cancer from 
mammograms, identifying diabetic retinopathy from retinal scans, and spotting pneumonia in 
chest X-rays.

Key Applications:
1. Medical Imaging: AI systems analyze CT scans, MRIs, and X-rays to detect abnormalities
2. Drug Discovery: ML accelerates identification of potential drug candidates by 40%
3. Clinical Decision Support: AI assists physicians in treatment planning
4. Predictive Analytics: Models predict patient readmission and disease progression
5. Administrative Automation: Natural language processing streamlines documentation

The global AI in healthcare market is projected to reach $45.2 billion by 2026, growing at 
44.9% CAGR. Major challenges include data privacy concerns, regulatory compliance, algorithm 
bias, and integration with existing systems.

Recent advancements include GPT-4 passing medical licensing exams and FDA-approved AI 
diagnostic tools reaching over 500 as of 2024. However, experts emphasize that AI should 
augment rather than replace human clinical judgment.""",
        "metadata": {
            "source": "ai_healthcare_report.pdf",
            "title": "AI in Healthcare Report 2024",
            "author": "Healthcare AI Institute",
            "date": "2024"
        }
    },
    "climate_policy": {
        "content": """Climate Change Policy Analysis: Global Perspectives

Climate change represents one of the most pressing challenges of our time. International 
efforts centered around the Paris Agreement aim to limit global warming to 1.5°C above 
pre-industrial levels. As of 2024, 195 countries have committed to nationally determined 
contributions (NDCs).

Key Policy Mechanisms:
1. Carbon Pricing: 46 countries have implemented carbon taxes or cap-and-trade systems
2. Renewable Energy Mandates: Over 170 countries have renewable energy targets
3. Green Finance: Climate-aligned investments reached $1.3 trillion in 2023
4. Regulatory Standards: Emissions standards for vehicles, buildings, and industry
5. International Cooperation: Climate funds support developing nations

The European Union leads with its Green Deal, targeting climate neutrality by 2050. China, 
the world's largest emitter, pledged carbon neutrality by 2060. The United States rejoined 
Paris Agreement and set 50% emissions reduction target by 2030.

Challenges include balancing economic development with emissions reduction, technology 
transfer to developing nations, and ensuring just transition for workers in fossil fuel 
industries. Scientists emphasize the need for more ambitious targets to meet 1.5°C goal.""",
        "metadata": {
            "source": "climate_policy_2024.pdf",
            "title": "Global Climate Policy Analysis",
            "author": "Environmental Policy Institute",
            "date": "2024"
        }
    },
    "financial_analysis": {
        "content": """Q4 2024 Financial Market Analysis

Global financial markets experienced significant volatility in Q4 2024, driven by geopolitical 
tensions, central bank policy shifts, and evolving economic conditions. The S&P 500 gained 
8.5% for the quarter, while emerging markets showed mixed performance.

Key Trends:
1. Interest Rates: Federal Reserve maintained rates at 5.25-5.5% with hints of 2025 cuts
2. Inflation: US CPI moderated to 3.1%, approaching but not reaching 2% target
3. Currency Markets: Dollar index fell 2.3% amid rate expectations
4. Commodities: Oil prices stabilized around $75/barrel despite OPEC+ cuts
5. Tech Sector: AI-related stocks led gains with 45% average increase

Regional Performance:
- US: S&P 500 +8.5%, NASDAQ +12.1%, Dow Jones +6.2%
- Europe: STOXX 600 +4.3%, struggling with energy costs
- Asia: Nikkei +7.8%, Shanghai Composite -1.2%
- Emerging Markets: MSCI EM +3.5%

Outlook for 2025: Analysts project moderate growth with potential Fed rate cuts supporting 
asset prices. Key risks include inflation persistence, geopolitical conflicts, and commercial 
real estate stress. Consensus earnings growth forecast: 11% for S&P 500.""",
        "metadata": {
            "source": "q4_financial_report.pdf",
            "title": "Q4 2024 Market Analysis",
            "author": "Global Finance Research",
            "date": "2024"
        }
    },
    "software_architecture": {
        "content": """Modern Software Architecture: Patterns and Best Practices

Software architecture continues to evolve with cloud-native and microservices patterns 
dominating enterprise development. This guide covers essential architectural concepts for 
building scalable, maintainable systems.

Core Patterns:
1. Microservices: Decompose applications into independently deployable services
2. Event-Driven: Use events for loose coupling between components
3. CQRS: Separate read and write operations for optimized data access
4. Domain-Driven Design: Align software with business domain models
5. Hexagonal Architecture: Isolate core logic from external dependencies

Cloud-Native Principles:
- Containerization: Docker and Kubernetes for deployment consistency
- Service Mesh: Istio, Linkerd for inter-service communication
- Observability: Distributed tracing, logging, metrics (OpenTelemetry)
- Infrastructure as Code: Terraform, Pulumi for reproducible environments
- GitOps: Declarative configuration and automated deployments

Best Practices:
- Design for failure with circuit breakers and retry patterns
- Implement proper authentication and authorization (OAuth 2.0, OIDC)
- Use API gateways for traffic management and security
- Apply 12-factor app principles
- Maintain comprehensive documentation and API contracts

Anti-Patterns to Avoid:
- Distributed monolith: Microservices with tight coupling
- Over-engineering: Premature optimization and complexity
- Ignoring operational concerns: DevOps and SRE considerations essential""",
        "metadata": {
            "source": "software_architecture_guide.pdf",
            "title": "Software Architecture Guide 2024",
            "author": "Tech Architecture Council",
            "date": "2024"
        }
    },
    "legal_contract": {
        "content": """Standard Software Licensing Agreement Summary

This document summarizes key provisions of enterprise software licensing agreements and 
important legal considerations for technology contracts.

License Types:
1. Perpetual License: One-time purchase with indefinite use rights
2. Subscription License: Time-limited access with recurring fees
3. Usage-Based: Pricing based on consumption metrics
4. Site License: Unlimited users at specified locations
5. Open Source: Various licenses (MIT, Apache, GPL) with different obligations

Key Contract Terms:
- Intellectual Property: Licensee receives limited use rights, not ownership
- Warranties: Typically limited to material conformance with documentation
- Liability Caps: Usually limited to fees paid in preceding 12 months
- Indemnification: Vendor indemnifies against IP infringement claims
- Data Protection: Must comply with GDPR, CCPA, and applicable privacy laws

Important Considerations:
- Audit Rights: Vendors may verify compliance with license terms
- Assignment: Review restrictions on transferring licenses
- Termination: Understand exit provisions and data return obligations
- SLA Terms: Define uptime guarantees and remedies for service failures
- Change of Control: Impact of mergers/acquisitions on license rights

Best Practices:
- Conduct thorough due diligence before signing
- Negotiate favorable limitation of liability terms
- Ensure clear data ownership and portability rights
- Include appropriate confidentiality provisions""",
        "metadata": {
            "source": "legal_licensing_summary.pdf",
            "title": "Software Licensing Legal Guide",
            "author": "Technology Law Associates",
            "date": "2024"
        }
    }
}


# 20+ Test Cases
TEST_CASES: List[TestCase] = [
    # === Basic Retrieval (Easy) ===
    TestCase(
        name="simple_fact_query",
        description="Basic factual question answering from a single document",
        difficulty="easy",
        category="retrieval",
        query="What is the projected market size for AI in healthcare by 2026?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["$45.2 billion", "2026", "healthcare"],
        expected_citations=1,
        timeout_seconds=30
    ),
    TestCase(
        name="multi_document_search",
        description="Retrieve information across multiple documents",
        difficulty="easy",
        category="retrieval",
        query="What are the key trends mentioned in the documents?",
        documents=[
            SAMPLE_DOCUMENTS["ai_healthcare"],
            SAMPLE_DOCUMENTS["financial_analysis"]
        ],
        expected_topics=["AI", "healthcare", "financial", "market"],
        expected_citations=2,
        timeout_seconds=45
    ),
    TestCase(
        name="keyword_search",
        description="Find specific information using keywords",
        difficulty="easy",
        category="retrieval",
        query="Explain carbon pricing mechanisms in climate policy",
        documents=[SAMPLE_DOCUMENTS["climate_policy"]],
        expected_topics=["carbon", "pricing", "cap-and-trade", "tax"],
        expected_citations=1,
        timeout_seconds=30
    ),
    
    # === Analysis Tasks (Medium) ===
    TestCase(
        name="pattern_detection",
        description="Identify patterns across documents",
        difficulty="medium",
        category="analysis",
        query="What patterns can you identify in how different sectors are adopting AI?",
        documents=[
            SAMPLE_DOCUMENTS["ai_healthcare"],
            SAMPLE_DOCUMENTS["financial_analysis"]
        ],
        expected_topics=["AI", "adoption", "pattern", "trend"],
        expected_citations=2,
        timeout_seconds=60
    ),
    TestCase(
        name="comparative_analysis",
        description="Compare and contrast information from multiple sources",
        difficulty="medium",
        category="analysis",
        query="Compare the regulatory approaches to AI in healthcare vs software licensing",
        documents=[
            SAMPLE_DOCUMENTS["ai_healthcare"],
            SAMPLE_DOCUMENTS["legal_contract"]
        ],
        expected_topics=["regulation", "compliance", "FDA", "legal"],
        expected_citations=2,
        timeout_seconds=60
    ),
    TestCase(
        name="trend_identification",
        description="Identify and explain trends from data",
        difficulty="medium",
        category="analysis",
        query="What market trends are evident from the Q4 2024 financial data?",
        documents=[SAMPLE_DOCUMENTS["financial_analysis"]],
        expected_topics=["trend", "market", "growth", "performance"],
        expected_citations=1,
        timeout_seconds=45
    ),
    TestCase(
        name="cause_effect_analysis",
        description="Analyze cause and effect relationships",
        difficulty="medium",
        category="analysis",
        query="What factors are driving AI adoption in healthcare?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["factor", "driving", "adoption", "AI", "healthcare"],
        expected_citations=1,
        timeout_seconds=45
    ),
    
    # === Complex Synthesis (Hard) ===
    TestCase(
        name="multi_source_report",
        description="Generate comprehensive report from multiple sources",
        difficulty="hard",
        category="synthesis",
        query="Create a comprehensive analysis of technology trends across healthcare, finance, and software development based on the provided documents",
        documents=[
            SAMPLE_DOCUMENTS["ai_healthcare"],
            SAMPLE_DOCUMENTS["financial_analysis"],
            SAMPLE_DOCUMENTS["software_architecture"]
        ],
        expected_topics=["technology", "trend", "healthcare", "finance", "software"],
        expected_citations=3,
        timeout_seconds=90
    ),
    TestCase(
        name="executive_summary",
        description="Generate executive-level summary of complex information",
        difficulty="hard",
        category="synthesis",
        query="Provide an executive summary of global climate policy progress and challenges",
        documents=[SAMPLE_DOCUMENTS["climate_policy"]],
        expected_topics=["climate", "policy", "Paris", "emissions"],
        expected_citations=1,
        timeout_seconds=60
    ),
    TestCase(
        name="recommendation_generation",
        description="Generate actionable recommendations based on analysis",
        difficulty="hard",
        category="synthesis",
        query="Based on current trends, what recommendations would you make for a healthcare organization considering AI adoption?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["recommendation", "AI", "healthcare", "implementation"],
        expected_citations=1,
        timeout_seconds=60
    ),
    
    # === Edge Cases (Various) ===
    TestCase(
        name="ambiguous_query",
        description="Handle ambiguous or vague queries gracefully",
        difficulty="hard",
        category="edge_case",
        query="Tell me about the thing with the numbers",
        documents=[SAMPLE_DOCUMENTS["financial_analysis"]],
        expected_topics=[],  # Should handle gracefully
        expected_citations=0,
        timeout_seconds=45
    ),
    TestCase(
        name="contradictory_sources",
        description="Handle potentially contradictory information",
        difficulty="hard",
        category="edge_case",
        query="What is the recommended approach for software architecture?",
        documents=[SAMPLE_DOCUMENTS["software_architecture"]],
        expected_topics=["architecture", "microservices", "pattern"],
        expected_citations=1,
        timeout_seconds=45
    ),
    TestCase(
        name="missing_information",
        description="Gracefully handle queries with incomplete information",
        difficulty="medium",
        category="edge_case",
        query="What is the healthcare AI market size in Japan specifically?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=[],  # Should acknowledge missing info
        expected_citations=0,
        timeout_seconds=30
    ),
    
    # === Robustness (Easy-Medium) ===
    TestCase(
        name="malformed_input",
        description="Handle malformed or unusual input",
        difficulty="easy",
        category="robustness",
        query="   What    is  AI???   ",  # Extra spaces, punctuation
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["AI"],
        expected_citations=1,
        timeout_seconds=30
    ),
    TestCase(
        name="empty_corpus",
        description="Handle query with no matching documents",
        difficulty="easy",
        category="robustness",
        query="What are the quantum computing applications?",
        documents=[],  # No documents
        expected_topics=[],
        expected_citations=0,
        timeout_seconds=30
    ),
    TestCase(
        name="very_long_query",
        description="Handle unusually long query input",
        difficulty="medium",
        category="robustness",
        query="I would like to understand the comprehensive overview of artificial intelligence applications in the healthcare sector, including but not limited to medical imaging analysis, drug discovery acceleration, clinical decision support systems, predictive analytics for patient outcomes, and administrative automation through natural language processing, with specific attention to market projections, growth rates, regulatory considerations, and the balance between AI capabilities and human clinical judgment",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["AI", "healthcare"],
        expected_citations=1,
        timeout_seconds=60
    ),
    
    # === Domain-Specific (Hard) ===
    TestCase(
        name="technical_analysis",
        description="Deep technical analysis of software concepts",
        difficulty="hard",
        category="domain",
        query="Explain the trade-offs between microservices and monolithic architectures based on the documentation",
        documents=[SAMPLE_DOCUMENTS["software_architecture"]],
        expected_topics=["microservices", "architecture", "pattern", "trade-off"],
        expected_citations=1,
        timeout_seconds=60
    ),
    TestCase(
        name="financial_summary",
        description="Financial data analysis and summarization",
        difficulty="hard",
        category="domain",
        query="Provide a quantitative summary of the Q4 2024 market performance with specific numbers",
        documents=[SAMPLE_DOCUMENTS["financial_analysis"]],
        expected_topics=["S&P", "NASDAQ", "percentage", "growth"],
        expected_citations=1,
        timeout_seconds=60
    ),
    TestCase(
        name="legal_review",
        description="Legal document analysis and interpretation",
        difficulty="hard",
        category="domain",
        query="What are the key liability and indemnification considerations in software licensing?",
        documents=[SAMPLE_DOCUMENTS["legal_contract"]],
        expected_topics=["liability", "indemnification", "contract", "legal"],
        expected_citations=1,
        timeout_seconds=60
    ),
    TestCase(
        name="medical_research",
        description="Medical/health research analysis",
        difficulty="hard",
        category="domain",
        query="What are the FDA-approved AI applications in medical diagnosis?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["FDA", "AI", "diagnosis", "approved"],
        expected_citations=1,
        timeout_seconds=60
    ),
    TestCase(
        name="scientific_literature",
        description="Scientific analysis and interpretation",
        difficulty="hard",
        category="domain",
        query="What scientific evidence supports the 1.5°C climate target?",
        documents=[SAMPLE_DOCUMENTS["climate_policy"]],
        expected_topics=["1.5°C", "climate", "target", "Paris"],
        expected_citations=1,
        timeout_seconds=60
    ),
    
    # === Multi-Step Reasoning (Hard) ===
    TestCase(
        name="chain_of_thought",
        description="Complex reasoning requiring multiple steps",
        difficulty="hard",
        category="reasoning",
        query="If AI adoption in healthcare continues at the projected rate and regulatory approvals accelerate, what might be the implications for medical practice by 2030?",
        documents=[SAMPLE_DOCUMENTS["ai_healthcare"]],
        expected_topics=["AI", "healthcare", "future", "implication"],
        expected_citations=1,
        timeout_seconds=90
    ),
    TestCase(
        name="hypothesis_testing",
        description="Evaluate a hypothesis against available evidence",
        difficulty="hard",
        category="reasoning",
        query="Evaluate the hypothesis: 'Cloud-native architecture is always the best choice for new software projects'",
        documents=[SAMPLE_DOCUMENTS["software_architecture"]],
        expected_topics=["cloud", "architecture", "microservices"],
        expected_citations=1,
        timeout_seconds=60
    ),
]


def get_test_cases_by_category(category: str) -> List[TestCase]:
    """Get test cases filtered by category.
    
    Args:
        category: Category to filter by.
        
    Returns:
        List[TestCase]: Filtered test cases.
    """
    return [tc for tc in TEST_CASES if tc.category == category]


def get_test_cases_by_difficulty(difficulty: str) -> List[TestCase]:
    """Get test cases filtered by difficulty.
    
    Args:
        difficulty: Difficulty level to filter by.
        
    Returns:
        List[TestCase]: Filtered test cases.
    """
    return [tc for tc in TEST_CASES if tc.difficulty == difficulty]


def get_all_sample_documents() -> List[Dict[str, Any]]:
    """Get all sample documents for testing.
    
    Returns:
        List[Dict]: All sample documents.
    """
    return list(SAMPLE_DOCUMENTS.values())
