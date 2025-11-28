"""Fact Checker Tool for SmartDoc Analyst.

This tool verifies claims against source documents
and provides confidence scores for factual assertions.
"""

from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult


class FactCheckerTool(BaseTool):
    """Fact verification tool for checking claims against sources.
    
    Analyzes claims and verifies them against provided source
    documents, returning confidence scores and supporting evidence.
    
    Attributes:
        llm: Language model for semantic analysis.
        confidence_threshold: Minimum confidence for verified claims.
        
    Example:
        >>> tool = FactCheckerTool(llm=my_llm)
        >>> result = await tool.execute(
        ...     claim="AI adoption increased by 50% in 2024",
        ...     sources=[{"content": "..."}]
        ... )
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        confidence_threshold: float = 0.7
    ):
        """Initialize the fact checker tool.
        
        Args:
            llm: Language model for analysis.
            confidence_threshold: Minimum confidence threshold.
        """
        super().__init__(
            name="fact_checker",
            description="Verify claims against source documents"
        )
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Verify a claim against source documents.
        
        Args:
            claim: The claim to verify.
            sources: List of source documents.
            
        Returns:
            ToolResult: Verification result with confidence.
        """
        claim = kwargs.get("claim", "")
        sources = kwargs.get("sources", [])
        
        if not claim:
            return ToolResult(
                success=False,
                error="Claim is required"
            )
            
        try:
            result = await self._verify_claim(claim, sources)
            
            return ToolResult(
                success=True,
                data=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Verification failed: {str(e)}"
            )
            
    async def verify(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convenience method for verification.
        
        Args:
            claim: Claim to verify.
            sources: Source documents.
            
        Returns:
            Dict: Verification result.
        """
        result = await self.execute(claim=claim, sources=sources)
        return result.data if result.success else {}
        
    async def _verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal claim verification logic.
        
        Args:
            claim: Claim to verify.
            sources: Source documents.
            
        Returns:
            Dict: Verification result.
        """
        if not sources:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "reason": "No source documents provided"
            }
            
        # Extract source content
        source_texts = []
        for source in sources:
            if isinstance(source, dict):
                content = source.get("content", source.get("text", ""))
            else:
                content = str(source)
            if content:
                source_texts.append(content)
                
        if not source_texts:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": [],
                "reason": "No content found in sources"
            }
            
        # Use LLM for sophisticated verification if available
        if self.llm:
            return await self._llm_verify(claim, source_texts)
        else:
            return self._heuristic_verify(claim, source_texts)
            
    async def _llm_verify(
        self,
        claim: str,
        source_texts: List[str]
    ) -> Dict[str, Any]:
        """Verify claim using LLM analysis.
        
        Args:
            claim: Claim to verify.
            source_texts: Source document texts.
            
        Returns:
            Dict: Verification result.
        """
        combined_sources = "\n\n---\n\n".join(source_texts[:3])[:6000]
        
        prompt = f"""Fact-check the following claim against the provided sources.

CLAIM: {claim}

SOURCES:
{combined_sources}

Analyze whether the claim is supported by the sources.
Respond in this exact format:
VERIFIED: [YES/NO/PARTIAL]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [Quote or paraphrase supporting evidence if found]
REASON: [Brief explanation]"""

        try:
            response = await self.llm.generate(prompt)
            return self._parse_llm_response(response, claim)
        except Exception:
            return self._heuristic_verify(claim, source_texts)
            
    def _parse_llm_response(
        self,
        response: str,
        claim: str
    ) -> Dict[str, Any]:
        """Parse LLM verification response.
        
        Args:
            response: LLM response text.
            claim: Original claim.
            
        Returns:
            Dict: Parsed verification result.
        """
        result = {
            "claim": claim,
            "verified": False,
            "confidence": 0.0,
            "evidence": [],
            "reason": ""
        }
        
        if not response:
            result["reason"] = "No response from verifier"
            return result
            
        lines = response.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("VERIFIED:"):
                value = line.replace("VERIFIED:", "").strip().upper()
                if "YES" in value:
                    result["verified"] = True
                elif "PARTIAL" in value:
                    result["verified"] = True  # Partial counts as verified
                    
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.replace("CONFIDENCE:", "").strip())
                    result["confidence"] = min(max(conf, 0.0), 1.0)
                except ValueError:
                    pass
                    
            elif line.startswith("EVIDENCE:"):
                evidence = line.replace("EVIDENCE:", "").strip()
                if evidence and evidence.lower() != "none":
                    result["evidence"].append(evidence)
                    
            elif line.startswith("REASON:"):
                result["reason"] = line.replace("REASON:", "").strip()
                
        return result
        
    def _heuristic_verify(
        self,
        claim: str,
        source_texts: List[str]
    ) -> Dict[str, Any]:
        """Verify claim using heuristic text matching.
        
        Args:
            claim: Claim to verify.
            source_texts: Source document texts.
            
        Returns:
            Dict: Verification result.
        """
        claim_lower = claim.lower()
        claim_words = set(
            word.strip(".,!?") for word in claim_lower.split()
            if len(word) > 3
        )
        
        evidence = []
        max_overlap = 0.0
        best_source_idx = -1
        
        for idx, source in enumerate(source_texts):
            source_lower = source.lower()
            
            # Calculate word overlap
            source_words = set(
                word.strip(".,!?") for word in source_lower.split()
                if len(word) > 3
            )
            
            if not claim_words:
                continue
                
            overlap = len(claim_words & source_words) / len(claim_words)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_source_idx = idx
                
            # Check for key phrases from claim in source
            claim_phrases = self._extract_key_phrases(claim)
            for phrase in claim_phrases:
                if phrase.lower() in source_lower:
                    # Find the sentence containing the phrase
                    sentences = source.split(".")
                    for sentence in sentences:
                        if phrase.lower() in sentence.lower():
                            evidence.append(sentence.strip()[:200])
                            break
                            
        # Calculate confidence
        confidence = min(max_overlap, 0.95)
        
        # Boost confidence if evidence found
        if evidence:
            confidence = min(confidence + 0.2, 0.95)
            
        verified = confidence >= self.confidence_threshold
        
        return {
            "claim": claim,
            "verified": verified,
            "confidence": round(confidence, 2),
            "evidence": evidence[:3],  # Limit evidence items
            "reason": f"Word overlap: {max_overlap:.0%}, Evidence found: {len(evidence)}"
        }
        
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text.
        
        Args:
            text: Text to extract phrases from.
            
        Returns:
            List[str]: Key phrases.
        """
        phrases = []
        words = text.split()
        
        # Extract 2-3 word phrases
        for i in range(len(words) - 1):
            phrase = " ".join(words[i:i+2])
            if len(phrase) > 5:
                phrases.append(phrase)
                
            if i < len(words) - 2:
                phrase = " ".join(words[i:i+3])
                phrases.append(phrase)
                
        # Extract numbers and percentages
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        phrases.extend(numbers)
        
        return phrases[:10]  # Limit phrases
        
    async def batch_verify(
        self,
        claims: List[str],
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify multiple claims against sources.
        
        Args:
            claims: List of claims to verify.
            sources: Source documents.
            
        Returns:
            List[Dict]: Verification results for each claim.
        """
        results = []
        for claim in claims:
            result = await self.execute(claim=claim, sources=sources)
            if result.success:
                results.append(result.data)
            else:
                results.append({
                    "claim": claim,
                    "verified": False,
                    "confidence": 0.0,
                    "error": result.error
                })
        return results
        
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters.
        
        Returns:
            Dict: Parameter schema.
        """
        return {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim to verify"
                },
                "sources": {
                    "type": "array",
                    "description": "Source documents for verification",
                    "items": {
                        "type": "object"
                    }
                }
            },
            "required": ["claim"]
        }
