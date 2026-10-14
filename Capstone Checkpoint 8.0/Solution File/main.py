"""
main.py
Launcher for the safe retrieval agent and graph diagnosis tests.

Developer: Gaurav Singh
Date: 2026-10-13


Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       python -m pip install --upgrade pip
       pip install -r requirements.txt   # pinned versions — avoids dependency-drift errors
2. Add the OpenRouter API key provided for this program. Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Wiki data: place the wiki files (one .txt per page) in a folder
   named 'WikiFiles' in this directory, or pass a folder path as the first
   argument. Filenames must follow wiki_<page>.txt so the date tool works.
   The folder MUST contain .txt files. The vector DB is persisted to ./chroma_db.

Running this file replays two Lab 6.1 attacks (the fake-wikifile injection and the
"chicken" roleplay) against the hardened agent so you can see it RESIST them, after running a
fast offline self-check that the two input filters (angle-bracket escape + wikifile-header strip) neutralize the payloads.
"""





import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WIKIFILES_DIR_DEFAULT = "WikiFiles"
WIKIFILES_TEST_DIR_DEFAULT = "WikiFiles_Test"

def require_api_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "Create a .env file in this folder containing:\n"
            "    OPENROUTER_API_KEY=sk-or-your-key-here\n"
        )


def _run_script(script_name: str, args: list[str], label: str) -> None:
    script_path = Path(__file__).resolve().with_name(script_name)
    if not script_path.is_file():
        print(f"\n[error] Could not find {script_name} at {script_path}")
        return

    try:
        subprocess.run(
            [sys.executable, str(script_path), *args],
            check=True,
            cwd=str(script_path.parent),
        )
    except subprocess.CalledProcessError as exc:
        print(f"\n[error] {label} exited with code {exc.returncode}.")
        print("Check the API key, dependencies, input files, and WikiFiles directory.")


def run_retrieval_mode(wikifiles_dir: str) -> None:
    """Launch the safe agent's interactive chat mode."""
    print("\n### Retrieval mode: interactive chat ###")
    _run_script(
        "retriever.py",
        [wikifiles_dir, "--chat"],
        "Safe agent",
    )


def run_testing_mode(wikifiles_dir: str) -> None:
    """Run the graph diagnosis evaluation against testInputs.json."""
    script_path = Path(__file__).resolve().with_name(
        "retrieval_Graph_Diagnosis.py"
    )
    inputs_path = script_path.with_name("testInputs.json")
    print("\n### Testing mode: graph diagnosis evaluation ###")
    _run_script(
        script_path.name,
        ["graph", wikifiles_dir, "--inputs", str(inputs_path)],
        "Graph diagnosis evaluation",
    )


def main() -> None:
    require_api_key()
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    wikifiles_dir = args[0] if args else WIKIFILES_DIR_DEFAULT

    while True:
        print("\nSelect an option:")
        print("1. Retrieval")
        print("2. Testing")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            run_retrieval_mode(wikifiles_dir)
        elif choice == "2":
            run_testing_mode(WIKIFILES_TEST_DIR_DEFAULT)
        elif choice == "3":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Please select a number from 1 to 3.")


if __name__ == "__main__":
    main()
