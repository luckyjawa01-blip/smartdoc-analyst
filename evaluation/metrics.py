"""Evaluation metrics for SmartDoc Analyst.

This module provides metrics calculation for system evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.
    
    Attributes:
        task_success_rate: Percentage of tasks completed successfully.
        completeness_score: Coverage of expected topics (0-1).
        relevance_score: Query-response alignment (0-1).
        coherence_score: Structural quality (0-1).
        citation_accuracy: Source attribution quality (0-1).
        response_latency_ms: End-to-end response time.
        token_efficiency: Tokens per successful query.
        hallucination_rate: Rate of false information (0-1).
        user_satisfaction: Simulated user rating (0-10).
    """
    task_success_rate: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    citation_accuracy: float = 0.0
    response_latency_ms: float = 0.0
    token_efficiency: float = 0.0
    hallucination_rate: float = 0.0
    user_satisfaction: float = 0.0
    
    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)
    
    def overall_score(self) -> float:
        """Calculate overall weighted score.
        
        Returns:
            float: Overall score (0-1).
        """
        weights = {
            "task_success_rate": 0.20,
            "completeness_score": 0.15,
            "relevance_score": 0.20,
            "coherence_score": 0.10,
            "citation_accuracy": 0.10,
            "hallucination_penalty": 0.15,  # Inverted
            "user_satisfaction_normalized": 0.10
        }
        
        score = (
            weights["task_success_rate"] * self.task_success_rate +
            weights["completeness_score"] * self.completeness_score +
            weights["relevance_score"] * self.relevance_score +
            weights["coherence_score"] * self.coherence_score +
            weights["citation_accuracy"] * self.citation_accuracy +
            weights["hallucination_penalty"] * (1 - self.hallucination_rate) +
            weights["user_satisfaction_normalized"] * (self.user_satisfaction / 10)
        )
        
        return round(score, 3)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.
        
        Returns:
            Dict: Metrics as dictionary.
        """
        return {
            "task_success_rate": self.task_success_rate,
            "completeness_score": self.completeness_score,
            "relevance_score": self.relevance_score,
            "coherence_score": self.coherence_score,
            "citation_accuracy": self.citation_accuracy,
            "response_latency_ms": self.response_latency_ms,
            "token_efficiency": self.token_efficiency,
            "hallucination_rate": self.hallucination_rate,
            "user_satisfaction": self.user_satisfaction,
            "overall_score": self.overall_score(),
            "details": self.details
        }


def calculate_metrics(
    query: str,
    response: str,
    expected_topics: List[str],
    expected_citations: int,
    sources_found: List[Dict[str, Any]],
    execution_time_ms: float,
    success: bool
) -> EvaluationMetrics:
    """Calculate evaluation metrics for a single response.
    
    Args:
        query: Original query.
        response: Generated response.
        expected_topics: Topics expected in response.
        expected_citations: Minimum expected citations.
        sources_found: Sources cited in response.
        execution_time_ms: Response generation time.
        success: Whether the query succeeded.
        
    Returns:
        EvaluationMetrics: Calculated metrics.
    """
    metrics = EvaluationMetrics()
    
    # Task success
    metrics.task_success_rate = 1.0 if success else 0.0
    
    # Completeness - check topic coverage
    if expected_topics:
        response_lower = response.lower()
        topics_found = sum(
            1 for topic in expected_topics
            if topic.lower() in response_lower
        )
        metrics.completeness_score = topics_found / len(expected_topics)
    else:
        metrics.completeness_score = 1.0 if response else 0.0
        
    # Relevance - check query-response alignment
    metrics.relevance_score = calculate_relevance(query, response)
    
    # Coherence - check structural quality
    metrics.coherence_score = calculate_coherence(response)
    
    # Citation accuracy
    if expected_citations > 0:
        actual_citations = len(sources_found) if sources_found else 0
        metrics.citation_accuracy = min(actual_citations / expected_citations, 1.0)
    else:
        metrics.citation_accuracy = 1.0
        
    # Response latency
    metrics.response_latency_ms = execution_time_ms
    
    # Token efficiency (approximation)
    tokens = len(response.split()) if response else 0
    metrics.token_efficiency = tokens
    
    # Hallucination rate (heuristic check)
    metrics.hallucination_rate = estimate_hallucination_rate(response, sources_found)
    
    # User satisfaction (simulated based on other metrics)
    metrics.user_satisfaction = simulate_user_satisfaction(metrics)
    
    return metrics


def calculate_relevance(query: str, response: str) -> float:
    """Calculate relevance score between query and response.
    
    Args:
        query: Original query.
        response: Generated response.
        
    Returns:
        float: Relevance score (0-1).
    """
    if not query or not response:
        return 0.0
        
    # Extract significant words from query
    query_words = set(
        word.lower().strip(".,!?")
        for word in query.split()
        if len(word) > 3
    )
    
    if not query_words:
        return 0.5
        
    # Check presence in response
    response_lower = response.lower()
    matches = sum(1 for word in query_words if word in response_lower)
    
    base_score = matches / len(query_words)
    
    # Bonus for longer, substantive responses
    word_count = len(response.split())
    if word_count > 100:
        base_score = min(base_score + 0.1, 1.0)
    if word_count > 200:
        base_score = min(base_score + 0.1, 1.0)
        
    return round(base_score, 3)


def calculate_coherence(response: str) -> float:
    """Calculate coherence score for response structure.
    
    Args:
        response: Generated response.
        
    Returns:
        float: Coherence score (0-1).
    """
    if not response:
        return 0.0
        
    score = 0.5  # Base score
    
    # Check for proper structure
    
    # Has paragraphs
    paragraphs = [p for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        score += 0.1
        
    # Has headers/sections
    if "##" in response or re.search(r'^#+ ', response, re.MULTILINE):
        score += 0.1
        
    # Has lists
    if "- " in response or re.search(r'^\d+\.', response, re.MULTILINE):
        score += 0.1
        
    # Has proper sentence structure
    sentences = re.split(r'[.!?]+', response)
    if len(sentences) >= 3:
        score += 0.1
        
    # Has logical connectors
    connectors = ["therefore", "however", "moreover", "additionally", 
                  "furthermore", "in conclusion", "as a result", "because"]
    if any(c in response.lower() for c in connectors):
        score += 0.1
        
    return min(round(score, 3), 1.0)


def estimate_hallucination_rate(
    response: str,
    sources: List[Dict[str, Any]]
) -> float:
    """Estimate hallucination rate in response.
    
    Args:
        response: Generated response.
        sources: Source documents used.
        
    Returns:
        float: Estimated hallucination rate (0-1).
    """
    if not response:
        return 0.0
        
    if not sources:
        # Can't verify without sources
        return 0.3  # Default uncertainty
        
    # Combine source content
    source_content = " ".join(
        s.get("content", "") for s in sources
    ).lower()
    
    if not source_content:
        return 0.3
        
    # Check for specific claims
    # Look for numbers/statistics
    numbers = re.findall(r'\$?\d+(?:\.\d+)?(?:%|billion|million)?', response)
    unsupported_numbers = 0
    
    for num in numbers:
        if num not in source_content:
            unsupported_numbers += 1
            
    if numbers:
        number_hallucination = unsupported_numbers / len(numbers)
    else:
        number_hallucination = 0.0
        
    # Check for strong claims
    strong_claims = ["always", "never", "definitely", "certainly", "proven"]
    strong_claim_count = sum(1 for c in strong_claims if c in response.lower())
    
    # Combine estimates
    rate = (number_hallucination * 0.7) + (min(strong_claim_count * 0.1, 0.3))
    
    return round(min(rate, 1.0), 3)


def simulate_user_satisfaction(metrics: EvaluationMetrics) -> float:
    """Simulate user satisfaction based on other metrics.
    
    Args:
        metrics: Current evaluation metrics.
        
    Returns:
        float: Simulated satisfaction (0-10).
    """
    # Weighted combination of metrics
    satisfaction = (
        metrics.task_success_rate * 3.0 +
        metrics.completeness_score * 2.0 +
        metrics.relevance_score * 2.5 +
        metrics.coherence_score * 1.5 +
        (1 - metrics.hallucination_rate) * 1.0
    )
    
    return round(min(satisfaction, 10.0), 1)


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
    """Aggregate multiple metrics into summary statistics.
    
    Args:
        metrics_list: List of metrics to aggregate.
        
    Returns:
        EvaluationMetrics: Aggregated metrics.
    """
    if not metrics_list:
        return EvaluationMetrics()
        
    n = len(metrics_list)
    
    return EvaluationMetrics(
        task_success_rate=sum(m.task_success_rate for m in metrics_list) / n,
        completeness_score=sum(m.completeness_score for m in metrics_list) / n,
        relevance_score=sum(m.relevance_score for m in metrics_list) / n,
        coherence_score=sum(m.coherence_score for m in metrics_list) / n,
        citation_accuracy=sum(m.citation_accuracy for m in metrics_list) / n,
        response_latency_ms=sum(m.response_latency_ms for m in metrics_list) / n,
        token_efficiency=sum(m.token_efficiency for m in metrics_list) / n,
        hallucination_rate=sum(m.hallucination_rate for m in metrics_list) / n,
        user_satisfaction=sum(m.user_satisfaction for m in metrics_list) / n,
        details={
            "sample_count": n,
            "individual_scores": [m.overall_score() for m in metrics_list]
        }
    )
