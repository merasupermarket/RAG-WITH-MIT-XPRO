"""Lab 4.1 (SOLUTION) — multistep retrieval by query decomposition.

Developer: Gaurav Singh, 09/12/2026
git- https://github.com/merasupermarket/RAG-WITH-MIT-XPRO

Run:  python lab_4_1_multistep_retrieval_solution.py "WikiFiles"

A single user question often contains several distinct information needs. Here the
model first rewrites the question into a few focused sub-queries. Then each sub-query
is run through the hybrid retriever (Lab 2.2), and the results are merged so that
an WikiFiles matching several sub-queries rises to the top. This approach is useful for questions that require information
assembled from different kinds of WikiFiles.

Pipeline:  question -> decompose into 2-5 sub-queries -> hybrid.getTopK(3*k) each
           -> merge by summing each WikiFiles's score across sub-queries -> top-k.

Setup
-----
1. Ensure that you have completed the program's one-time environment and
   OpenRouter API key setup, and activate the configured environment.

2. Install the required dependencies, if they are not already available:
       pip install -r requirements.txt

3. Locate the WikiFiles data: Ensure that the provided .txt WikiFiles are
   available in the 'WikiFiles' folder, or pass the folder path when
   running the script.

Sample questions to try (over the WikiFiles corpus)
-----------------------------------------------
Multipart questions benefit most because each part becomes its own sub-query:
    "what was Apollo 11? who approved it? and when was it launched?"
        -> [decompose] 'what was Apollo 11? who approved it? and when was it launched?' ->
            1. Apollo 11 mission overview and objectives
            2. Approval authority and decision-makers for the Apollo 11 mission
            3. Apollo 11 launch date and launch site details
            Assistant: **What was Apollo 11?**  
            Apollo 11 was the American crewed spaceflight that first landed humans on the Moon. It was the fifth crewed mission of NASA’s Apollo program, 
            a “crewed lunar landing (G)” that carried a three‑person crew (Neil Armstrong, Michael Collins and Edwin “Buzz” Aldrin) and lasted 8 days 3 hours 18 minutes 35 seconds.

            **Who approved it?**  
            The supplied WikiFiless do not name a specific individual or body that formally approved the Apollo 11 mission. The documents describe the mission’s planning, 
            goals set by President John F. Kennedy, and the NASA teams that executed it, but they do not record a definitive approval authority.
            **When was it launched?**  
            Apollo 11 launched on **16 July 1969 at 13:32:00 UTC** (9:32 am EDT) aboard the Saturn V SA‑506 from Launch Complex 39A at Kennedy Space Center.

    "who initiated the print of coins to celebrate the Apollo 11 and on which anniversary?"
        -> The initiative came from **Congressman Bill Posey**, who in June 2015 introduced resolution H.R. 2726 directing the 
            United States Mint to design and sell commemorative gold, silver and clad coins **for the 50th‑anniversary of the Apollo 11 mission**."
"""
import json
import os
import re
import sys
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 4
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
DEBUG = True  # Print the sub-queries the model produces

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a company WikiFiles database. "
    "Break the user's question into 2-5 focused retrieval sub-queries that together "
    "cover everything needed to answer it. Each sub-query should target a distinct "
    "aspect of the question; you may include a paraphrase of the original as one "
    "sub-query. Phrase each as if searching for specific WikiFiles. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

SYSTEM_PROMPT = """You are a helpful assistant for Precision Paperclip Inc. \
You answer questions by drawing information exclusively from the company WikiFiles \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided WikiFiles, answer clearly and concisely.
- If the provided WikiFiles do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided WikiFiles."""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Use the OpenRouter API key provided for this program.\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


def chat_loop(response):
    print("Chat over the WikiFiles DB. Type your question; 'exit'/'quit' to stop.\n")
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


class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._history = [SystemMessage(content=SYSTEM_PROMPT)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str: ...

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (WikiFiles):\n{context}\n\nQuestion: {query}"

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


# ─── Keyword + vector + hybrid stack (from Modules 1-2) ──────────────
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
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(WikiFiles: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print("Building vector DB (first run — embedding the WikiFiles)...")
    docs = []
    for fname in sorted(os.listdir(WikiFiles)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(WikiFiles, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} WikiFiles into {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


def _load_wikifiles(WikiFiles: str) -> tuple[list[str], list[str]]:
    paths = sorted(os.path.join(WikiFiles, f) for f in os.listdir(WikiFiles) if f.endswith(".txt"))
    contents = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            contents.append(fh.read())
    return [os.path.basename(p) for p in paths], contents


class HybridRetriever(BaseRetriever):
    def __init__(self, WikiFiles: str, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._paths, self._contents = _load_wikifiles(WikiFiles)
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        self._db = db

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        bm_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:CANDIDATE_POOL]
        bm = [(self._paths[i], self._contents[i], scores[i]) for i in bm_idx]
        vec = [
            (d.metadata.get("source", "unknown"), d.page_content, s)
            for d, s in self._db.similarity_search_with_score(query, k=CANDIDATE_POOL)
        ]
        content_by_name, bm_norm, vec_norm = {}, {}, {}
        for (n, c, _), v in zip(bm, _normalize([s for _, _, s in bm])):
            content_by_name[n] = c
            bm_norm[n] = v
        for (n, c, _), v in zip(vec, _normalize([d for _, _, d in vec], invert=True)):
            content_by_name[n] = c
            vec_norm[n] = v
        fused = [
            (n, c, WEIGHT_BM25 * bm_norm.get(n, 0.0) + WEIGHT_VECTOR * vec_norm.get(n, 0.0))
            for n, c in content_by_name.items()
        ]
        fused.sort(key=lambda t: t[2], reverse=True)
        return fused[:k]

    def retrievedContext(self, query: str) -> str:
        return "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c, _ in self.getTopK(query, NUM_RETRIEVED))


# ─── Multi-step retrieval ────────────────────────────────────────────
class MultiStepRetriever(BaseRetriever):
    def __init__(self, hybrid: HybridRetriever, top_k: int = NUM_RETRIEVED, debug: bool = DEBUG, **kwargs):
        super().__init__(**kwargs)
        self._hybrid = hybrid
        self._top_k = top_k
        self._debug = debug

    def _decompose_query(self, query: str) -> list[str]:
        """Ask the LLM to split the question into focused sub-queries. Returns a list
        of strings. On any malformed response, it falls back to [query] so the
        pipeline degrades to plain hybrid retrieval instead of crashing."""
        messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM),
            HumanMessage(content=query),
        ]
        response = self._llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw.strip()
        if raw.startswith("```"):  # Tolerate ```json ... ``` fences
            raw = raw.strip("`")
            raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [query]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Decompose the query, retrieve 3*k candidates per sub-query with the hybrid
        retriever, and merge by summing each WikiFiles's score across the sub-queries
        that retrieved it. An WikiFiles matching multiple sub-queries ranks higher."""
        sub_queries = self._decompose_query(query)
        if self._debug:
            print(f"[decompose] {query!r} ->")
            for i, sq in enumerate(sub_queries, 1):
                print(f"   {i}. {sq}")
        content_map: dict[str, str] = {}
        score_map: dict[str, float] = {}
        for sq in sub_queries:
            for name, content, score in self._hybrid.getTopK(sq, 3 * k):
                content_map[name] = content
                score_map[name] = score_map.get(name, 0.0) + score
        top = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(name, content_map[name], score) for name, score in top]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, self._top_k)
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)


if __name__ == "__main__":
    require_api_key()
    WikiFiles = sys.argv[1] if len(sys.argv) > 1 else "WikiFiles"
    db = build_or_load_db(WikiFiles, CHROMA_DIR)
    hybrid = HybridRetriever(WikiFiles=WikiFiles, db=db)
    MultiStepRetriever(hybrid=hybrid).chat()