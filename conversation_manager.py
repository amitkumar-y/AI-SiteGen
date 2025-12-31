"""
Conversation Manager Module
Handles multi-turn conversation chains and context preservation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationMessage:
    """Represents a single message in a conversation."""

    def __init__(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationChain:
    """Manages conversation history and context for multi-turn dialogues."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[ConversationMessage] = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(role, content, metadata))

        # Trim history if needed
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]

    def get_context_for_llm(self, include_last_n: int = 3) -> List[Dict[str, str]]:
        """Get formatted conversation context for LLM."""
        recent_messages = self.messages[-(include_last_n * 2):]
        return [{"role": msg.role, "content": msg.content} for msg in recent_messages]

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message content."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_proposals(self) -> Optional[List[Dict]]:
        """Get proposals from the last assistant response."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and "proposals" in msg.metadata:
                return msg.metadata["proposals"]
        return None

    def has_previous_context(self) -> bool:
        """Check if there's previous conversation context."""
        return len(self.messages) > 0

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation."""
        if not self.messages:
            return ""

        user_queries = [msg.content for msg in self.messages if msg.role == "user"]
        if not user_queries:
            return ""

        summary = ["Previous requests:"]
        for i, query in enumerate(user_queries, 1):
            summary.append(f"  {i}. {query}")

        return "\n".join(summary)

    def detect_follow_up(self, current_query: str) -> bool:
        """Detect if the current query is a follow-up question."""
        follow_up_patterns = [
            "change", "modify", "update", "make it", "darker", "lighter",
            "more", "less", "different", "instead", "also", "add", "remove",
            "replace", "that", "this", "the first", "the second", "option",
        ]

        query_lower = current_query.lower()

        # Short queries likely reference previous context
        if len(current_query.split()) < 5 and self.has_previous_context():
            return True

        # Check for follow-up keywords
        return any(pattern in query_lower for pattern in follow_up_patterns)

    def build_contextual_query(self, current_query: str) -> str:
        """Build a contextual query by combining current query with history."""
        if not self.has_previous_context():
            return current_query

        if self.detect_follow_up(current_query):
            last_query = self.get_last_user_message()
            if last_query:
                return f"Context from previous request: {last_query}\n\nCurrent request: {current_query}"

        return current_query

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.messages),
            "messages": [msg.to_dict() for msg in self.messages],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "conversation_id": self.conversation_id,
            "total_turns": len(self.messages) // 2,
            "user_messages": sum(1 for msg in self.messages if msg.role == "user"),
            "assistant_messages": sum(1 for msg in self.messages if msg.role == "assistant"),
            "has_context": self.has_previous_context(),
        }


def create_conversation_chain(max_history: int = 10) -> ConversationChain:
    """Factory function to create a conversation chain."""
    return ConversationChain(max_history=max_history)
