"""
Model Manager Module
--------------------
Handles automatic LLM model switching with fallback logic.
When one model reaches capacity, automatically switches to the next available model.

Supports multiple Groq models with intelligent failover.
"""

import os
from typing import Optional, Dict, Any, List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class ModelManager:
    """
    Manages multiple LLM models with automatic fallback.
    Switches between models when rate limits or capacity issues occur.
    """

    def __init__(self):
        """Initialize the model manager with Groq API."""
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

        self.client = Groq(api_key=self.api_key)

        # Available Groq models (all free)
        # Ordered by preference: fastest to most capable
        self.models = [
            {
                "name": "llama-3.1-8b-instant",
                "description": "Fast, efficient model for quick responses",
                "max_tokens": 8192,
                "active": True
            },
            {
                "name": "llama-3.3-70b-versatile",
                "description": "More capable model for complex tasks",
                "max_tokens": 32768,
                "active": True
            },
            {
                "name": "mixtral-8x7b-32768",
                "description": "High-quality model with large context",
                "max_tokens": 32768,
                "active": True
            }
        ]

        self.current_model_index = 0
        self.retry_attempts = {}  # Track retry attempts per model

        print(f"[OK] Model Manager initialized with {len(self.models)} models")
        print(f"    Primary: {self.models[0]['name']}")

    def get_current_model(self) -> str:
        """Get the name of the current active model."""
        return self.models[self.current_model_index]["name"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return self.models[self.current_model_index].copy()

    def switch_to_next_model(self) -> bool:
        """
        Switch to the next available model in the list.

        Returns:
            bool: True if switched successfully, False if no more models available
        """
        original_index = self.current_model_index

        # Try next models in sequence
        for i in range(1, len(self.models)):
            next_index = (self.current_model_index + i) % len(self.models)

            if self.models[next_index]["active"]:
                old_model = self.models[self.current_model_index]["name"]
                self.current_model_index = next_index
                new_model = self.models[self.current_model_index]["name"]

                print(f"[SWITCH] Model switched: {old_model} -> {new_model}")
                return True

        # No alternative models available
        print(f"[WARN] No alternative models available. Staying with {self.get_current_model()}")
        return False

    def reset_to_primary(self) -> None:
        """Reset to the primary (first) model."""
        if self.current_model_index != 0:
            old_model = self.get_current_model()
            self.current_model_index = 0
            new_model = self.get_current_model()
            print(f"[RESET] Model reset to primary: {old_model} -> {new_model}")

    def is_capacity_error(self, error: Exception) -> bool:
        """
        Check if an error is related to model capacity/rate limits.

        Args:
            error: The exception to check

        Returns:
            bool: True if it's a capacity/rate limit error
        """
        error_message = str(error).lower()

        capacity_indicators = [
            "rate limit",
            "rate_limit",
            "too many requests",
            "capacity",
            "overloaded",
            "quota",
            "429",
            "503",
            "service unavailable"
        ]

        return any(indicator in error_message for indicator in capacity_indicators)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> Any:
        """
        Make a chat completion request with automatic model fallback.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_retries: Maximum number of retry attempts across all models

        Returns:
            Groq API response object

        Raises:
            Exception: If all models fail after retries
        """
        attempts = 0
        last_error = None
        models_tried = []

        while attempts < max_retries:
            current_model = self.get_current_model()
            models_tried.append(current_model)

            try:
                # Build request parameters
                request_params = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add response format if specified
                if response_format:
                    request_params["response_format"] = response_format

                # Make the API call
                response = self.client.chat.completions.create(**request_params)

                # Success - reset to primary model for next request
                if self.current_model_index != 0 and attempts > 0:
                    self.reset_to_primary()

                return response

            except Exception as e:
                last_error = e
                attempts += 1

                print(f"[ERROR] Model {current_model} failed (attempt {attempts}/{max_retries})")
                print(f"        Error: {str(e)[:100]}")

                # Check if it's a capacity error
                if self.is_capacity_error(e):
                    # Try switching to next model
                    if attempts < max_retries:
                        switched = self.switch_to_next_model()
                        if not switched:
                            # No more models to try, but we have retry attempts left
                            # Reset to primary and try again
                            self.reset_to_primary()
                else:
                    # Not a capacity error, don't retry
                    raise e

        # All retries exhausted
        models_str = ", ".join(set(models_tried))
        error_msg = f"All models failed after {max_retries} attempts. Models tried: {models_str}. Last error: {str(last_error)}"
        print(f"[FATAL] {error_msg}")
        raise Exception(error_msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage."""
        return {
            "total_models": len(self.models),
            "active_models": sum(1 for m in self.models if m["active"]),
            "current_model": self.get_current_model(),
            "current_model_index": self.current_model_index,
            "models_list": [m["name"] for m in self.models if m["active"]]
        }


# Singleton instance
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get or create the singleton ModelManager instance.

    Returns:
        ModelManager: The singleton instance
    """
    global _model_manager_instance

    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()

    return _model_manager_instance


if __name__ == "__main__":
    # Test the model manager
    print("=" * 60)
    print("Testing Model Manager")
    print("=" * 60)

    # Create manager
    manager = get_model_manager()

    # Show stats
    print("\n1. Initial Stats:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test model switching
    print("\n2. Testing model switching:")
    for i in range(5):
        print(f"\n   Switch {i+1}:")
        print(f"   Current: {manager.get_current_model()}")
        manager.switch_to_next_model()

    # Reset to primary
    print("\n3. Reset to primary:")
    manager.reset_to_primary()
    print(f"   Current: {manager.get_current_model()}")

    # Test actual API call (if API key is available)
    print("\n4. Testing actual API call:")
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' in exactly 3 words."}
        ]

        response = manager.chat_completion(
            messages=test_messages,
            temperature=0.7,
            max_tokens=50
        )

        print(f"   Success! Model used: {response.model}")
        print(f"   Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"   API call failed: {e}")

    print("\n[OK] Model manager testing complete!")
