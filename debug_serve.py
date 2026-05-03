"""Debug entry point - hardcodes CLI args for debugging."""
import sys
sys.argv = [
    "serve.py",
    "--model", "Qwen/Qwen3-0.6B",
    "--dtype", "bf16",
    "--max_length", "2048",
    "--device", "cuda",
]

from serve import main
main()
