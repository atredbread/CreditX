"""
Example usage of the prompt logging system.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.prompt_logger import log_prompt, get_prompt_history
from src.prompt_decorator import log_prompts

# Example 1: Basic logging
print("Logging example prompts...")
log_prompt("What is the current project structure?")
log_prompt("Can we log all the prompts that I am giving you?")

# Example 2: Using the decorator
@log_prompts
def process_prompt(prompt: str) -> str:
    """Example function that processes a prompt."""
    return f"Processed: {prompt}"

# This call will be automatically logged
result = process_prompt("This is a test prompt")
print(f"Processing result: {result}")

# View recent prompt history
print("\nRecent prompt history:")
history = get_prompt_history()
for i, entry in enumerate(history, 1):
    print(f"{i}. {entry['timestamp']} - {entry['prompt']}")

print("\nPrompt logging example complete. Check the 'logs/prompt_history.log' file for the complete log.")
