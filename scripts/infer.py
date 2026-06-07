"""Interactive inference script - complete sentences using trained MiniGPT.

The model starts with your prompt and generates the rest of the sentence.

Usage:
    # Interactive REPL mode (type sentences to complete)
    python scripts/infer.py --corpus 100
    
    # Complete a single sentence
    python scripts/infer.py --corpus 100 "MAN RUN"
    
    # Batch completion from file (one prompt per line)
    python scripts/infer.py --corpus 100 --file prompts.txt
    
    # Adjust generation parameters
    python scripts/infer.py --corpus 100 --max-tokens 20 --temperature 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model.tokenizer import WordTokenizer
from src.model.transformer import MiniGPT

MODELS_DIR = PROJECT_ROOT / "data" / "models"
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "generated_texts"


def load_model(corpus_size: int, device: str = "cpu") -> tuple[MiniGPT, WordTokenizer]:
    """Load trained model and tokenizer from checkpoint."""
    corpus_path = CORPUS_DIR / f"generated_corpus_{corpus_size}.txt"
    ckpt_path = MODELS_DIR / f"minigpt_corpus{corpus_size}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train first with: python scripts/train/train_model.py --corpus {corpus_size}"
        )

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # Load tokenizer from saved vocab
    tokenizer = WordTokenizer.from_corpus(corpus_path)
    
    # Load model (always use CPU to avoid CUDA compatibility issues)
    device = "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MiniGPT(vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[OK] Loaded model: {ckpt_path.name}")
    print(f"  Epoch {ckpt['epoch']}, Loss {ckpt['loss']:.4f}")
    print(f"[OK] Tokenizer: {tokenizer.vocab_size} tokens")
    print(f"[OK] Device: {device}")
    print()
    return model, tokenizer


def complete_sentence(
    model: MiniGPT,
    tokenizer: WordTokenizer,
    prompt: str,
    max_tokens: int = 20,
    temperature: float = 0.8,
    device: str = "cpu",
) -> str:
    """Complete a sentence given a prompt.
    
    Args:
        model: Trained MiniGPT model
        tokenizer: WordTokenizer
        prompt: Starting words (space-separated, e.g., "MAN RUN")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: CPU or CUDA
    
    Returns:
        Completed sentence (prompt + generated tokens)
    """
    # Encode prompt
    prompt_tokens = prompt.strip().split()
    if not prompt_tokens:
        raise ValueError("Prompt cannot be empty")
    
    # Check all words exist in vocab
    missing = [w for w in prompt_tokens if w not in tokenizer.token_to_id]
    if missing:
        return f"Unknown words: {', '.join(missing)}"
    
    # Encode and generate
    prompt_ids = [tokenizer.token_to_id[w] for w in prompt_tokens]
    
    # Add BOS at start (if not already present)
    if prompt_ids[0] != tokenizer.bos_id:
        prompt_ids = [tokenizer.bos_id] + prompt_ids
    
    # Generate continuation
    with torch.no_grad():
        # all_ids holds the full sequence: BOS + prompt + generated so far
        all_ids = list(prompt_ids)

        for _ in range(max_tokens):
            # Feed only the last context_len tokens to avoid exceeding model limit
            context = torch.tensor([all_ids[-model.context_len:]], device=device)
            logits = model(context)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == tokenizer.eos_id:
                break

            all_ids.append(next_id)

    # Decode — strip BOS, return full sentence (prompt + generated)
    output_ids = all_ids[1:]  # remove BOS
    return tokenizer.decode(output_ids)


def interactive_mode(model: MiniGPT, tokenizer: WordTokenizer, args: argparse.Namespace) -> None:
    """Interactive REPL for sentence completion."""
    device = "cpu"
    
    print("=" * 70)
    print("  Interactive Mode: Type a prompt and the model will complete it")
    print("=" * 70)
    print("Commands:")
    print("  quit           - Exit")
    print("  temp <value>   - Change temperature (default: 0.8, range 0.1-2.0)")
    print("  max <tokens>   - Change max tokens (default: 20)")
    print("=" * 70)
    print()

    temperature = args.temperature
    max_tokens = args.max_tokens
    
    while True:
        try:
            prompt = input("Prompt> ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == "quit":
                print("Goodbye!")
                break
            
            if prompt.lower().startswith("temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"Temperature set to {temperature}")
                except (IndexError, ValueError):
                    print("Usage: temp <value> (e.g., temp 0.9)")
                continue
            
            if prompt.lower().startswith("max "):
                try:
                    max_tokens = int(prompt.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except (IndexError, ValueError):
                    print("Usage: max <tokens> (e.g., max 25)")
                continue
            
            # Complete the sentence
            result = complete_sentence(
                model, tokenizer, prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )
            print(f"Result: {result}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            print()


def batch_mode(model: MiniGPT, tokenizer: WordTokenizer, input_file: str, args: argparse.Namespace) -> None:
    """Batch completion from file."""
    device = "cpu"
    
    file_path = Path(input_file)
    if not file_path.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    results = []
    with file_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            prompt = line.strip()
            if not prompt:
                continue
            
            result = complete_sentence(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )
            results.append((prompt, result))
            print(f"[{i:3d}] {result}")
    
    print(f"\n[OK] Completed {len(results)} sentences")


def main(args: argparse.Namespace) -> None:
    model, tokenizer = load_model(args.corpus, device="cpu")
    
    if args.prompt:
        # Single sentence mode
        result = complete_sentence(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device="cpu",
        )
        print(f"Completion: {result}")
        
    elif args.file:
        # Batch mode
        batch_mode(model, tokenizer, args.file, args)
        
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete sentences using trained MiniGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default — type sentences to complete)
  python scripts/infer.py --corpus 100
  
  # Complete a single sentence
  python scripts/infer.py --corpus 100 "MAN RUN FAST"
  
  # Batch: complete sentences from file (one per line)
  python scripts/infer.py --corpus 100 --file prompts.txt
  
  # Adjust parameters
  python scripts/infer.py --corpus 100 --max-tokens 25 --temperature 0.9
  
Commands in interactive mode:
  quit            - Exit
  temp <value>    - Change temperature (0.1-2.0)
  max <tokens>    - Change max tokens to generate
        """,
    )
    
    p.add_argument(
        "prompt", nargs="?", default=None,
        help="Sentence prompt to complete (optional; if omitted, enters interactive mode)"
    )
    p.add_argument(
        "--corpus", type=int, default=10000, choices=[100, 500, 1000, 5000, 10000],
        help="Corpus size the model was trained on (default: 10000)"
    )
    p.add_argument(
        "--max-tokens", type=int, default=20,
        help="Maximum tokens to generate (default: 20)"
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature: higher = more random, lower = more deterministic (default: 0.8)"
    )
    p.add_argument(
        "--file", type=str, default=None,
        help="Batch mode: read prompts from file (one per line)"
    )
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

