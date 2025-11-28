"""Safety guards for SmartDoc Analyst.

This module provides input validation, safety checks,
and rate limiting for the system.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set


@dataclass
class ValidationResult:
    """Result of input validation.
    
    Attributes:
        valid: Whether the input is valid.
        issues: List of validation issues found.
        sanitized: Sanitized version of input.
        risk_score: Risk score (0-1, higher is riskier).
    """
    valid: bool
    issues: List[str]
    sanitized: Optional[str] = None
    risk_score: float = 0.0


class SafetyGuard:
    """Safety guard for input validation and protection.
    
    Provides comprehensive safety checks including:
    - Injection attack detection
    - PII detection
    - Rate limiting
    - Content sanitization
    
    Attributes:
        max_input_length: Maximum allowed input length.
        rate_limit_rpm: Rate limit in requests per minute.
        pii_detection: Enable PII detection.
        
    Example:
        >>> guard = SafetyGuard(max_input_length=10000)
        >>> result = guard.validate_input("User query here")
        >>> if result.valid:
        ...     process(result.sanitized)
    """
    
    # Patterns for injection detection
    INJECTION_PATTERNS = [
        r"(?i)(ignore|disregard|forget).*?(previous|prior|above).*?(instructions?|prompts?)",
        r"(?i)you are now",
        r"(?i)pretend (to be|you are)",
        r"(?i)act as",
        r"(?i)(system|admin|root) (prompt|instruction|command)",
        r"(?i)jailbreak",
        r"(?i)bypass.*?(filter|safety|guard)",
        r"(?i)override.*?(system|instruction)",
        r"(?i)\[system\]",
        r"(?i)<<.*?>>",  # System-like brackets
    ]
    
    # PII patterns
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }
    
    # Dangerous content patterns
    DANGEROUS_PATTERNS = [
        r"(?i)(how to|instructions for).*(hack|attack|exploit|steal)",
        r"(?i)(create|make|build).*(bomb|weapon|malware|virus)",
        r"(?i)(harm|hurt|kill|injure).*person",
    ]
    
    def __init__(
        self,
        max_input_length: int = 10000,
        rate_limit_rpm: int = 60,
        pii_detection: bool = True,
        block_dangerous: bool = True
    ):
        """Initialize the safety guard.
        
        Args:
            max_input_length: Maximum input length.
            rate_limit_rpm: Rate limit per minute.
            pii_detection: Enable PII detection.
            block_dangerous: Block dangerous content.
        """
        self.max_input_length = max_input_length
        self.rate_limit_rpm = rate_limit_rpm
        self.pii_detection = pii_detection
        self.block_dangerous = block_dangerous
        
        # Rate limiting state
        self._request_times: Dict[str, List[datetime]] = defaultdict(list)
        
        # Compile patterns
        self._injection_patterns = [
            re.compile(p) for p in self.INJECTION_PATTERNS
        ]
        self._pii_patterns = {
            k: re.compile(p) for k, p in self.PII_PATTERNS.items()
        }
        self._dangerous_patterns = [
            re.compile(p) for p in self.DANGEROUS_PATTERNS
        ]
        
    def validate_input(self, text: str) -> ValidationResult:
        """Validate and sanitize input text.
        
        Args:
            text: Input text to validate.
            
        Returns:
            ValidationResult: Validation result with issues and sanitized text.
        """
        issues = []
        risk_score = 0.0
        
        if not text:
            return ValidationResult(
                valid=True,
                issues=[],
                sanitized="",
                risk_score=0.0
            )
            
        # Check length
        if len(text) > self.max_input_length:
            issues.append(f"Input exceeds maximum length ({self.max_input_length})")
            risk_score += 0.3
            
        # Check for injection attempts
        injection_found = self._check_injection(text)
        if injection_found:
            issues.extend(injection_found)
            risk_score += 0.5
            
        # Check for PII
        if self.pii_detection:
            pii_found = self._check_pii(text)
            if pii_found:
                issues.append(f"Potential PII detected: {', '.join(pii_found)}")
                risk_score += 0.2
                
        # Check for dangerous content
        if self.block_dangerous:
            dangerous = self._check_dangerous(text)
            if dangerous:
                issues.extend(dangerous)
                risk_score += 0.8
                
        # Sanitize input
        sanitized = self._sanitize(text)
        
        # Determine validity
        valid = risk_score < 0.5 and not any(
            "dangerous" in issue.lower() for issue in issues
        )
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            sanitized=sanitized,
            risk_score=min(risk_score, 1.0)
        )
        
    def _check_injection(self, text: str) -> List[str]:
        """Check for prompt injection attempts.
        
        Args:
            text: Input text.
            
        Returns:
            List[str]: List of injection issues found.
        """
        issues = []
        
        for pattern in self._injection_patterns:
            if pattern.search(text):
                issues.append("Potential prompt injection detected")
                break
                
        return issues
        
    def _check_pii(self, text: str) -> List[str]:
        """Check for personally identifiable information.
        
        Args:
            text: Input text.
            
        Returns:
            List[str]: Types of PII found.
        """
        found = []
        
        for pii_type, pattern in self._pii_patterns.items():
            if pattern.search(text):
                found.append(pii_type)
                
        return found
        
    def _check_dangerous(self, text: str) -> List[str]:
        """Check for dangerous content.
        
        Args:
            text: Input text.
            
        Returns:
            List[str]: Dangerous content issues.
        """
        issues = []
        
        for pattern in self._dangerous_patterns:
            if pattern.search(text):
                issues.append("Potentially dangerous content detected")
                break
                
        return issues
        
    def _sanitize(self, text: str) -> str:
        """Sanitize input text.
        
        Args:
            text: Input text.
            
        Returns:
            str: Sanitized text.
        """
        # Truncate to max length
        sanitized = text[:self.max_input_length]
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
        
    def sanitize_output(self, response: str) -> str:
        """Sanitize output response.
        
        Args:
            response: Response text to sanitize.
            
        Returns:
            str: Sanitized response.
        """
        sanitized = response
        
        # Remove any PII that might have leaked
        for pii_type, pattern in self._pii_patterns.items():
            sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)
            
        # Remove system-like instructions
        sanitized = re.sub(r'\[system\].*?\[/system\]', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
        
    def rate_limit(self, user_id: str) -> bool:
        """Check rate limit for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            bool: True if within rate limit, False if exceeded.
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        # Clean old requests
        self._request_times[user_id] = [
            t for t in self._request_times[user_id]
            if t > window_start
        ]
        
        # Check limit
        if len(self._request_times[user_id]) >= self.rate_limit_rpm:
            return False
            
        # Record request
        self._request_times[user_id].append(now)
        return True
        
    def get_rate_limit_status(self, user_id: str) -> Dict[str, Any]:
        """Get rate limit status for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            Dict: Rate limit status.
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        recent = [
            t for t in self._request_times.get(user_id, [])
            if t > window_start
        ]
        
        remaining = max(0, self.rate_limit_rpm - len(recent))
        
        return {
            "limit": self.rate_limit_rpm,
            "remaining": remaining,
            "used": len(recent),
            "reset_in_seconds": 60 - (now - min(recent)).seconds if recent else 0
        }
        
    def validate_document(self, document: Dict[str, Any]) -> ValidationResult:
        """Validate a document for ingestion.
        
        Args:
            document: Document with content and metadata.
            
        Returns:
            ValidationResult: Validation result.
        """
        issues = []
        risk_score = 0.0
        
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        # Validate content
        content_result = self.validate_input(content)
        issues.extend(content_result.issues)
        risk_score = max(risk_score, content_result.risk_score)
        
        # Validate metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                meta_result = self.validate_input(value)
                if not meta_result.valid:
                    issues.append(f"Invalid metadata field: {key}")
                    risk_score = max(risk_score, meta_result.risk_score * 0.5)
                    
        return ValidationResult(
            valid=risk_score < 0.5,
            issues=issues,
            sanitized=content_result.sanitized,
            risk_score=risk_score
        )


class ContentFilter:
    """Content filter for response generation.
    
    Filters generated content to ensure safety and appropriateness.
    """
    
    # Words/phrases to filter
    BLOCKED_PHRASES: Set[str] = {
        "as an ai",
        "i cannot",
        "i'm not able to",
        "i don't have access",
    }
    
    def __init__(self, strict_mode: bool = False):
        """Initialize content filter.
        
        Args:
            strict_mode: Enable strict filtering.
        """
        self.strict_mode = strict_mode
        
    def filter(self, text: str) -> str:
        """Filter generated text.
        
        Args:
            text: Text to filter.
            
        Returns:
            str: Filtered text.
        """
        filtered = text
        
        if self.strict_mode:
            # Remove blocked phrases
            for phrase in self.BLOCKED_PHRASES:
                filtered = re.sub(
                    re.escape(phrase),
                    "",
                    filtered,
                    flags=re.IGNORECASE
                )
                
        # Clean up extra whitespace
        filtered = ' '.join(filtered.split())
        
        return filtered
        
    def is_safe(self, text: str) -> bool:
        """Check if text is safe to return.
        
        Args:
            text: Text to check.
            
        Returns:
            bool: True if safe.
        """
        guard = SafetyGuard()
        result = guard.validate_input(text)
        return result.valid
