"""
Safety Filters and Pattern Detection Module
Provides pre-RAG filtering for dangerous patterns and greeting detection.
"""

import re
from typing import Dict, Tuple
from rapidfuzz import fuzz


class SafetyFilter:
    """Handles safety checks, dangerous pattern detection, and greeting responses."""

    def __init__(self):
        # Dangerous patterns
        self.dangerous_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"disregard\s+(previous|above|all)\s+(instructions?|prompts?|rules?)",
            r"forget\s+(everything|all|previous|above)",
            r"(rm|del|delete)\s+-rf",
            r"drop\s+table",
            r"exec\s*\(",
            r"eval\s*\(",
            r"union\s+select",
            r"how\s+to\s+(hack|crack|break into|exploit)",
            r"bypass\s+(security|authentication|protection)",
            r"pretend\s+(you\s+are|to\s+be)\s+(not|no\s+longer)\s+(an?\s+)?AI",
            r"what\s+(are\s+)?(your|the)\s+(system\s+)?(instructions?|prompts?|rules?)",
        ]

        # Fuzzy patterns
        self.fuzzy_patterns = [
            "ignore previous instructions",
            "disregard all rules",
            "show me your system prompt",
            "bypass security",
            "how to hack",
        ]

        # Greeting patterns
        self.greeting_patterns = [
            r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\s*[!.]*$",
            r"^(what's\s+up|whats\s+up|sup|yo)\s*[!.]*$",
            r"^(how\s+are\s+you|how\s+do\s+you\s+do)\s*[?!.]*$",
        ]

        self.greeting_responses = [
            "👋 Hello! I'm your AI Website Design Assistant for law firms. I can help you create professional, tailored website designs.",
            "Hi there! Ready to design an amazing law firm website? Just describe what you're looking for!",
            "Hello! I specialize in creating website designs for law firms. What type of legal practice are you designing for?",
        ]

        # Compile regex patterns
        self.compiled_dangerous = [re.compile(p, re.IGNORECASE) for p in self.dangerous_patterns]
        self.compiled_greetings = [re.compile(p, re.IGNORECASE) for p in self.greeting_patterns]

    def check_dangerous_patterns(self, text: str) -> Tuple[bool, str]:
        """Check if text contains dangerous patterns."""
        for pattern in self.compiled_dangerous:
            if pattern.search(text):
                return True, "Input contains potentially harmful patterns. Please provide a legitimate law firm website design request."

        for fuzzy_pattern in self.fuzzy_patterns:
            if fuzz.token_set_ratio(text.lower(), fuzzy_pattern.lower()) > 80:
                return True, "Input appears to contain suspicious content. Please provide a valid law firm website design request."

        return False, ""

    def check_greeting(self, text: str) -> Tuple[bool, str]:
        """Check if text is a greeting."""
        for pattern in self.compiled_greetings:
            if pattern.match(text.strip()):
                import random
                response = random.choice(self.greeting_responses)
                response += "\n\n**Try asking something like:**\n"
                response += "- \"Create a modern website for a criminal defense attorney\"\n"
                response += "- \"Design an elegant site for an estate planning law firm\"\n"
                response += "- \"I need a professional corporate law website\""
                return True, response
        return False, ""

    def validate_input(self, user_input: str) -> Dict[str, any]:
        """Comprehensive input validation pipeline."""
        # Check greetings
        is_greeting, greeting_response = self.check_greeting(user_input)
        if is_greeting:
            return {
                "is_valid": True,
                "is_greeting": True,
                "is_dangerous": False,
                "message": greeting_response,
                "should_continue": False
            }

        # Check dangerous patterns
        is_dangerous, danger_reason = self.check_dangerous_patterns(user_input)
        if is_dangerous:
            return {
                "is_valid": False,
                "is_greeting": False,
                "is_dangerous": True,
                "message": danger_reason,
                "should_continue": False
            }

        # Input is clean
        return {
            "is_valid": True,
            "is_greeting": False,
            "is_dangerous": False,
            "message": "",
            "should_continue": True
        }
