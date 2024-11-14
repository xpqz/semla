import os
import sys
import anthropic
from typing import Optional, List, Dict, Generator
from enum import Enum
from pathlib import Path

class ClaudeModel(Enum):
    """Available Claude models with their identifiers."""
    OPUS = "claude-3-opus-20240229"
    SONNET = "claude-3-sonnet-20240229"
    HAIKU = "claude-3-haiku-20240307"

    @classmethod
    def list_models(cls) -> List[str]:
        """Return list of available model names."""
        return [model.name for model in cls]

class ClaudeClient:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: ClaudeModel = ClaudeModel.HAIKU,
                 system_prompt_file: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 4096):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Client(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history: List[Dict[str, str]] = []
        
        # Default APL-focused system prompt
        default_apl_prompt = """You are an expert in Dyalog APL programming particularly skilled in array-oriented thinking and tacit programming. You're a helpful assistant, that returns concise answers.
        
You understand APL's core principles of working with arrays as a whole rather than element-by-element operations.

Always bear in mind that APL is strictly evaluated right to left. Functions have long right scope, and short left scope.

When writing APL code, you strictly follow this grammar for dfn-style APL:

Key elements:
- Functions are defined using blocks: {body}
- Arguments are ⍺ (left) and ⍵ (right)
- Statements are separated by ⋄ or newlines
- Guards use : for conditional execution. A triggered guard always returns a value from the function
- Dfns return the value of the FIRST non-assignment statement.
- Variable names: lowercase for arrays, uppercase for functions
- Valid primitive functions: ∇⌷⍷⊆⊇⊂⊃!⍲⍱∩∪⊥⊤⊣⊢○~≢≡⍸↑↓∊⍉⊖⌽⍴⍳+×÷⍪,⌈|⌊-*/∨∧<>=≠≤≥⍋⍒?
- Valid monadic operators: ¨⌿⍀⍨⌸
- Valid dyadic operators: @⍥⍤⍣∘⍛
- Hybrids (glyphs that can be either a function or operator depending on context): ⌿⍀\/
- Numbers can include ¯ for negative and j for complex
- Strings are enclosed in single quotes with '' for escape
- System functions start with ⎕ (e.g., ⎕IO)

Examples of valid expressions:
- Simple dfn: {⍵+1}
- Guard usage: {⍵≤1: ⍵ ⋄ (∇⍵-1)+∇⍵-2}
- Array operations: +/⍳10
- Function assignment: Sum←+/
- +⌿⍳10                              ⍝ Sum first 10 integers
- {⍺←0 1 ⋄ ⍵=0:⍬⍴⍺ ⋄ (1↓⍺,+⌿⍺)∇ ⍵-1} ⍝ Tail-recursive Fibonacci
- Avg←+⌿÷≢                           ⍝ Named tacit function to calculate average

When writing code, ensure every part conforms to this grammar. Prefer clear, idiomatic solutions that leverage array operations and avoid explicit loops and recursion where possible.

When writing APL code snippets, write them directly without any delimiters, fences, or language markers. Just write the code itself.

When explaining code, you can use the ⍝ comment character at the end of lines for inline explanations.
"""
        
        # Load system prompt from file or use default
        self.system_prompt = ""
        if system_prompt_file:
            try:
                with open(system_prompt_file, 'r') as f:
                    self.system_prompt = f.read().strip()
                print(f"Loaded system prompt from {system_prompt_file}")
            except Exception as e:
                print(f"Warning: Could not load system prompt file: {e}")
                self.system_prompt = default_apl_prompt
                print("Using default APL system prompt")
        else:
            self.system_prompt = default_apl_prompt
            print("Using default APL system prompt")

    def stream_message(self, message: str) -> Generator[str, None, str]:
        """Stream a message to Claude and yield response chunks."""
        try:
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            with self.client.messages.stream(
                model=self.model.value,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=self.conversation_history
            ) as stream:
                full_response = ""
                
                for chunk in stream:
                    if chunk.type == "content_block_delta":
                        text = chunk.delta.text
                        full_response += text
                        yield text
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                return full_response
                
        except anthropic.APIError as e:
            error_msg = f"API Error: {str(e)}"
            yield error_msg
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            yield error_msg
            return error_msg
    
    def set_model(self, model: ClaudeModel):
        """Change the model being used."""
        self.model = model
        self.clear_history()
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

def main():
    """Example usage of the Claude client."""
    # Check for system prompt file argument
    system_prompt_file = None
    if len(sys.argv) > 1:
        system_prompt_file = sys.argv[1]
    
    client = ClaudeClient(system_prompt_file=system_prompt_file)
    
    print("Chat with Claude (type 'quit' to exit, 'clear' to reset conversation)")
    print("Available commands:")
    print("  'model' - show current model")
    print("  'models' - list available models")
    print("  'use model_name' - switch to a different model (e.g., 'use OPUS')")
    print("  'prompt' - show current system prompt")
    print("  'clear' - clear conversation history")
    print("------------------------------------------------")
    
    if client.system_prompt:
        print(f"\nLoaded system prompt ({len(client.system_prompt)} characters)")
        print("Use 'prompt' command to view it")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
            
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        elif user_input.lower() == 'clear':
            client.clear_history()
            print("\nConversation history cleared.")
            continue
        elif user_input.lower() == 'model':
            print(f"\nCurrent model: {client.model.name}")
            continue
        elif user_input.lower() == 'models':
            print("\nAvailable models:", ", ".join(ClaudeModel.list_models()))
            continue
        elif user_input.lower() == 'prompt':
            if client.system_prompt:
                print("\nSystem prompt:")
                print("-" * 40)
                print(client.system_prompt)
                print("-" * 40)
            else:
                print("\nNo system prompt loaded")
            continue
        elif user_input.lower().startswith('use '):
            try:
                model_name = user_input[4:].strip().upper()
                new_model = ClaudeModel[model_name]
                client.set_model(new_model)
                print(f"\nSwitched to model: {new_model.name}")
            except KeyError:
                print(f"\nInvalid model name. Available models: {', '.join(ClaudeModel.list_models())}")
            continue
        
        print("\nClaude:", end=" ", flush=True)
        for chunk in client.stream_message(user_input):
            print(chunk, end="", flush=True)
        print(f"\n\n(Using {client.model.name} model | Conversation history: {len(client.conversation_history)} messages)")

if __name__ == "__main__":
    main()