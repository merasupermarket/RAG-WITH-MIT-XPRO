"""Lab 1.2 (STARTER) — keyword retrieval over the company emails with BM25.

Run:  python lab_1_2_keyword_retrieval_starter.py <emails_dir>

Your job (see the Lab 1.2 walkthrough):
  - Step 1: Implement tokenize() — lowercase the input, split into tokens, and drop stopwords.
  - Step 2: Implement Bm25Retriever.getTopK() and retrievedContext().
The BaseRetriever base class, the chat loop, and the index loading are provided.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core rank-bm25
2. Add your OpenRouter API key (free at https://openrouter.ai/keys). Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Email data: Place the company email files (one .txt per email) in a folder
   named 'detailedEmails' in this directory, or pass a folder path as the first
   argument. The folder MUST contain the .txt files.

Sample questions to try (over the email corpus)
-----------------------------------------------
    "What was the status of the government project going into 2015?"
        -> Grounded answer: The project had been put on hold/paused.
    "Who was involved in discussions about restarting the government project?"
        -> Names people from the emails (e.g., the Finance Director and COO).
    "What is the CEO's home phone number?"
        -> This is NOT in the emails, so the assistant should refuse rather than guess.
Compare the retrievers on the same questions: Keyword (1.2) is strong on exact terms,
vector (2.1) handles paraphrases, and hybrid (2.2) combines both.
"""
import os
import re
import sys
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"  # free; "openai/gpt-4o-mini" (paid, cheap) also good
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company e-mails \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided e-mails, answer clearly and concisely.
- If the provided e-mails do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided e-mails."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Get a free key at https://openrouter.ai/keys\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


# ─── Provided: command-line chat loop ────────────────────────────────
def chat_loop(response):
    print("Chat over the email DB. Type your question; 'exit'/'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            result = response(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {result}\n")


# ─── Provided: the reusable retriever base class ─────────────────────
# Subclass it and implement retrievedContext(query); the base handles calling
# the LLM, conversation history, and the chat loop. (Reused in Modules 2-3.)
class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str:
        """Return a string of context retrieved for the given query."""

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (e-mails):\n{context}\n\nQuestion: {query}"

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(question, context)),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def queryWHistory(self, question: str) -> str:
        context = self.retrievedContext(question)
        self._history.append(HumanMessage(content=self._build_user_message(question, context)))
        try:
            response = self._llm.invoke(self._history)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception:
            self._history.pop()
            raise
        self._history.append(AIMessage(content=answer))
        return answer

    def chat(self) -> None:
        chat_loop(self.queryWHistory)


# ─── Keyword retrieval (your turn) ───────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}


def tokenize(text: str) -> list[str]:
    """Turn text into BM25 keywords. The SAME tokenizer indexes documents and
    tokenizes queries, so it must be consistent for both.

    TODO (Step 1): Lowercase the text, split into alphanumeric tokens, and delete
    the stopwords found in _STOPWORDS. Return the remaining tokens.
    (Hint: re.findall(r"[a-z0-9]+", text.lower()) gives lowercase tokens.)

    Delete the raise NotImplementedError line once your code works.
    """
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


class Bm25Retriever(BaseRetriever):
    def __init__(self, emails_dir: str, top_k: int = TOP_K, **kwargs):
        super().__init__(**kwargs)
        paths = sorted(
            os.path.join(emails_dir, f) for f in os.listdir(emails_dir) if f.endswith(".txt")
        )
        self._paths = [os.path.basename(p) for p in paths]
        self._contents = []
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as fh:
                self._contents.append(fh.read())
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        self._top_k = top_k
        print(f"Indexed {len(self._paths)} emails.")

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Return the k best-matching emails as (filename, content, score).

        TODO (Step 2): Tokenize the query, score every document with
        self._bm25.get_scores(tokens), and return the top-k as
        (self._paths[i], self._contents[i], scores[i]).

        Delete the raise NotImplementedError line once your code works.
        """
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:k]
        return [
            (self._paths[index], self._contents[index], scores[index])
            for index in ranked_indices
        ]

    def retrievedContext(self, query: str) -> str:
        """Join the top self._top_k results into one context string, each labeled
        '[filename]\\n<content>' and separated by dividers.

        TODO (Step 3). Delete the raise NotImplementedError line once it works.
        """
        #raise NotImplementedError("Implement retrievedContext() — see the TODO above.")
        results = self.getTopK(query, self._top_k)
        return "\n\n---\n\n".join(
            f"[{filename}]\n{content}" for filename, content, _ in results
        )

if __name__ == "__main__":
    require_api_key()
    emails_dir = sys.argv[1] if len(sys.argv) > 1 else "detailedEmails"
    Bm25Retriever(emails_dir=emails_dir).chat()