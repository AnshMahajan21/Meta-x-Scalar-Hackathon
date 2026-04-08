"""
server/app.py — OpenEnv entry point
Re-exports the FastAPI app and provides a main() for the server script.
"""

import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402

def main():
    """Entry point for `server` script defined in pyproject.toml."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
