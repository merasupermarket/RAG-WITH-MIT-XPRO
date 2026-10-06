r"""Lab 7.2 (SOLUTION) — Optimize performance and cost under load.

Developer: Gaurav Singh
Date: 2026-10-04

Module 7: Designing for Production — Cost, Security, and Reliability

Run:  python lab_7_2_efficient_agent_solution.py <wikifiles_dir>

A system can be CORRECT and SECURE and still fail in production if it is too slow or too
expensive under load. In Lab 6.2 we MEASURED the token cost of the agent and saw two facts:
(1) the full multi-step agent spends most of its tokens on the PLAN loop, and (2) only a
small minority of questions actually need that full pipeline — a cheap single-shot answer
is fine for the majority. This lab turns those measurements into two optimizations and then
MEASURES the saving against the un-optimized agent.

This is the SAME measured agent from Lab 6.2 (plan -> retrieve / by_date / clarify / answer,
with per-model token tracking) plus two production-oriented optimizations:

  1. Semantic cache (Step 1). A chromadb vector cache of (question -> answer). On a query we
     embed the question and look up the nearest cached entry; we accept it ONLY above a HIGH
     similarity threshold (CACHE_SIMILARITY_THRESHOLD = 0.85 — much higher than a retrieval
     threshold, because returning a wrong cached answer is worse than a cache miss). Because a
     chat question can be context-dependent ("when was he hired?"), on each cache hit we run
     a small model (_CACHE_VALIDATE_SYSTEM) to confirm the question stands on its own before
     trusting the cache. The store is bounded (CACHE_MAX_SIZE) and evicts the oldest entry.
     Note: the cache PERSISTS to CACHE_DIR between runs — delete that folder to re-measure the
     optimized cost from a cold cache (a warm cache makes repeat runs look near-free).
     For a stronger validation do BOTH: one cold run (delete CACHE_DIR first), then re-run the
     optimized experiment WITHOUT deleting it — the repeated questions should now show cache
     hits and a substantially lower second-run cost.

  2. Query router (Step 2). A cheap router model (ROUTER_MODEL = openai/gpt-5-nano,
     _ROUTER_SYSTEM) decides "direct" vs "agent". "direct" = one hybrid retrieval + a single
     small-model answer; if that answer is judged unsatisfactory (_SATISFACTION_SYSTEM) we fall
     back to the FULL agent on a stronger model (FALLBACK_MODEL = openai/gpt-5.4). Most simple
     questions never leave the cheap direct path; only the genuinely hard ones pay for the loop.

  3. Measure (Step 3). Keep Lab 6.2's per-model token instrumentation, run the test suite on
     the optimized agent AND on the un-optimized agent (cache + router disabled = the Lab 6.2
     agent), print per-model token usage + a rough total cost for each, and compare. Every model
     run is wrapped so one API failure (429 / 5xx) is SKIPPED with a message, never crashes the
     suite (see `_safe_experiment`).

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
3. WikiFiles data: place the company WikiFiles (one .txt per file) in a folder
   named 'WikiFiles' in this directory, or pass a folder path as the first
   argument. Filenames must follow wiki_MM_DD_YY_<id>.txt so the date tool works.
   The folder MUST contain .txt files. The vector DB is persisted to ./chroma_db and
   the semantic cache to ./semantic_cache.

Findings (illustrative — your exact numbers depend on the corpus and the run)
----------------------------------------------------------------------------
Running the small suite on the unoptimized agent, every question pays for at least one PLAN
call plus an ANSWER call (often several plan iterations), all on gpt-5.4-mini. With the router
on, most of these questions route "direct": a single retrieval + one small-model answer, gated
by a near-free gpt-5-nano router/satisfaction check — cutting the token bill for the common case
roughly in half, while the rare hard question still escalates to the full agent on gpt-5.4 so
quality is preserved. The semantic cache adds a second lever: re-asking the same (or a reworded)
question returns the stored answer for the price of one embedding lookup + one nano validation
call — near zero — which is exactly the "who is the CEO?" repeated-query case. Takeaway that ties
back to the CDF: optimization is not about one clever trick but about spending the expensive
model ONLY where it changes the answer, and not paying twice for a question you already answered.
You may need to tune the router/validation prompts on your own corpus before the cost drops.

IMPORTANT — cheaper is not automatically equivalent. Always compare the optimized answers
AGAINST the baseline answers, question by question, not just the two cost lines. The trade-off
this lab is designed to reveal is exactly that a mis-routed question or a lenient satisfaction
check can make the cheap path return a less complete answer at a lower price. Two safeguards in
this file address the failure modes we hit in testing: exact-date questions route "agent" (so
the deterministic by_date tool runs instead of near-date semantic matches), and the satisfaction
check sees the retrieved wikifiles (so an answer that omits entities they contain is escalated).
If the two paths still disagree materially on a question, verify against the source wikifiles
before trusting either.
"""
import glob
import json
import os
import re
import sys
import time
import uuid
from datetime import date
from typing import Optional, TypedDict
from AgentState import _AgentState

import chromadb
from dateutil import parser as dateutil_parser
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"        # default: plan, answer, and the direct path
ROUTER_MODEL = "openai/gpt-5-nano"       # lower-cost router + cache-validation + satisfaction checks
FALLBACK_MODEL = "openai/gpt-5.4"        # more capable model for the full agent when direct is unsatisfactory
EMBEDDING_MODEL = "openai/text-embedding-3-small"
CHROMA_DIR = "chroma_db"
CACHE_DIR = "semantic_cache"
CACHE_MAX_SIZE = 400
CACHE_SIMILARITY_THRESHOLD = 0.85        # HIGH on purpose — a wrong cached answer is worse than a miss
NUM_RETRIEVED = 5
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
MAX_ITERATIONS = 5
WIKIFILES_DIR_DEFAULT = "WikiFiles"

# ─── Cost experiment configuration (Step 3) ──────────────────
# A small test suite. Keep it brief because each question makes real API calls. Adapt to your corpus.
EXPERIMENT_QUESTIONS = [
    "Tell me something about Apollo 11?",
    "What was decided about the government project?",
    "Who was Nikita Khrushchev?",
    "Why did Armstrong refused to put their names on emblem?",
]


# APPROXIMATE prices in USD per 1,000,000 tokens — for a rough cost estimate ONLY.
# Verify live prices at https://openrouter.ai/models before quoting real numbers.
# (Rates are illustrative tier placeholders: nano < mini < the strong fallback model.)
APPROX_PRICE_PER_MTOK = {
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai/gpt-5.4-mini": {"input": 0.20, "output": 0.80},
    "openai/gpt-5.4": {"input": 2.50, "output": 10.0},
}

_FILENAME_RE = re.compile(r"^wiki_(\d{2})_(\d{2})_(\d{2})_\d+\.txt$")

ANSWER_SYSTEM = """You are a research assistant for a company WikiFiles.
Answer questions exclusively from the company wikifiles provided as context. Consider \
any clarifications, which may add important details; if the original message is not a \
full question, answer the last question asked in the clarifications. If the retrieved \
WikiFiles do not contain enough information, say so explicitly. Do not speculate or use \
outside knowledge. If the user is simply asking to exit, answer exactly: Exiting"""

PLAN_SYSTEM = """You are a research agent for Precision Paperclip Inc. You help the user \
find information about the company using a semantic search database of company WikiFiles.

You are given the user's question (and prior conversation), the queries already executed,
the WikiFiles retrieved so far, and any clarifications from the user. Decide what to do next
and respond with a JSON object ONLY:
{
  "action": "retrieve" | "by_date" | "clarify" | "answer",
  "queries": ["query1", "query2"],   // 1-3 NEW queries; only for action=="retrieve"; don't repeat executed ones
  "start_date": "MM/DD/YYYY",          // only for action=="by_date"
  "end_date": "MM/DD/YYYY",            // optional; only for action=="by_date" and a range is needed
  "clarification": "question text",    // only for action=="clarify"
  "reasoning": "brief explanation"
}

Guidelines:
- "retrieve": you need more information via semantic search.
- "by_date": the question references specific dates or a short range ("start of the month",
  "this week", "between X and Y"). Retrieves ALL WikiFiles on that day/range.
- "clarify": the message isn't really a question and you must ask for more input. Use ONLY
  as a last resort, and only AFTER attempting to query the database.
- "answer": you have enough information, or the user asked to exit.
Products are sometimes referred to by multiple names — searching alternate names can help."""

# ─── Router and cache prompts ──────────────────────────
# Design notes:
#  * ROUTER: Exact-date and date-range questions route to the full agent because the
#   direct path does not use the by_date tool. This avoids relying on semantically
#   similar WikiFiles that may come from the wrong date.
#  * SATISFACTION: The answer-quality check evaluates both usefulness and completeness
#   against the retrieved evidence, so answers that omit relevant people, products,
#   projects, or events can be escalated to the full agent.

_ROUTER_SYSTEM = """You are a query router for a company WikiFile research system.
Decide whether the question can be answered by a single semantic search over company WikiFiles,
or whether it requires the full multi-step agentic retrieval pipeline.

Route "direct" for: specific facts, people, or clear single-topic questions.
Route "agent" for: questions asking which WikiFiles were sent on an exact date or within a date range
(these require the deterministic by_date tool — semantic search alone retrieves near-date, wrong-date
WikiFiles), multi-step reasoning, synthesis across many WikiFiles, time-range analysis, or vague/ambiguous
questions.

Respond with ONLY valid JSON, no other text: {"route": "direct"} or {"route": "agent"}"""

_CACHE_VALIDATE_SYSTEM = """You are validating whether the question is a full question and the
answer is a proper response to the question. You should answer false if the question is incomplete or lacks context, even knowing that the
questions are all in the context of the company Precision Paperclip Inc.,
or if the answer is not related to the question. For example, a question like "What is her role?" is not a full question because
there is not enough context to know who "her" is.

Respond with ONLY valid JSON, no other text: {"valid": true} or {"valid": false, "reason": "reason for invalidity"}"""

_SATISFACTION_SYSTEM = """You are evaluating whether an answer satisfactorily addresses a question,
given the WikiFiles that were retrieved for it.
An answer is satisfactory only if it provides specific, useful information AND covers every
relevant entity (product, person, project, event) that the retrieved WikiFiles contain about
the question.
An answer is NOT satisfactory if it says the information is unavailable, is too vague, does not
address the question, or omits relevant entities that appear in the retrieved WikiFiles.

Respond with ONLY valid JSON, no other text: {"satisfactory": true} or {"satisfactory": false}"""


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
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


# ─── Hybrid retrieval stack (from Module 2 — unchanged from Lab 6.2) ──
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
        check_embedding_ctx_length=False,  # OpenRouter needs raw text, not pre-tokenized input
    )


def build_or_load_db(wikifiles_dir: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print("Building vector DB (first run — embedding the WikiFiles)...")
    docs = []
    for fname in sorted(os.listdir(wikifiles_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(wikifiles_dir, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} WikiFiles into {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)  # all-tied -> treat each result as equally relevant
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


def _load_wikifiles(wikifiles_dir: str) -> tuple[list[str], list[str]]:
    paths = sorted(os.path.join(wikifiles_dir, f) for f in os.listdir(wikifiles_dir) if f.endswith(".txt"))
    contents = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            contents.append(fh.read())
    return [os.path.basename(p) for p in paths], contents


class HybridRetriever:
    def __init__(self, wikifiles_dir: str, db: Chroma, retriever_model: Optional[str] = None):
        # retriever_model is accepted for parity with the agent's multi-model design. This
        # consolidated retriever is pure BM25 + embedding-vector fusion (fixed embedding
        # model) and issues NO chat-LLM calls, so it never appears in the token tally.
        self._retriever_model = retriever_model or LLM_MODEL
        self._paths, self._contents = _load_wikifiles(wikifiles_dir)
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


# ─── Semantic cache (Step 1) ─────────────────────────────────────────
class _SemanticCache:
    """Cosine-similarity cache of (question -> answer) pairs; evicts the oldest entry when full.

    We reuse the SAME embedding model as retrieval (get_embeddings, with the OpenRouter
    check_embedding_ctx_length=False flag) but look up at a MUCH higher threshold — returning a
    stale/wrong cached answer is worse than paying to recompute, so the bar for a "hit" is high.
    """

    def __init__(self, cache_dir: str = CACHE_DIR, max_size: int = CACHE_MAX_SIZE, debug: bool = False):
        self._client = chromadb.PersistentClient(path=cache_dir)
        self._col = self._client.get_or_create_collection(
            "response_cache",
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embeddings()  # ctx flag lives in get_embeddings()
        self._max_size = max_size
        self._debug = debug

    def lookup(self, question: str, threshold: float) -> Optional[tuple[str, float]]:
        """Return (cached_answer, similarity) if a close-enough match exists, else None."""
        count = self._col.count()
        if count == 0:
            if self._debug:
                print("[cache] empty")
            return None
        embedding = self._embedder.embed_query(question)
        results = self._col.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["metadatas", "distances"],
        )
        if not results["ids"][0]:
            return None
        distance = results["distances"][0][0]
        similarity = 1.0 - distance   # cosine space: distance = 1 - similarity
        if self._debug:
            nearest_q = results["metadatas"][0][0].get("question", "?")[:80]
            print(f"[cache] nearest similarity={similarity:.4f} q={nearest_q!r}")
        if similarity < threshold:
            if self._debug:
                print(f"[cache] miss — {similarity:.4f} < threshold {threshold:.4f}")
            return None
        if self._debug:
            print(f"[cache] hit — similarity={similarity:.4f}")
        return results["metadatas"][0][0]["answer"], similarity

    def store(self, question: str, answer: str) -> None:
        embedding = self._embedder.embed_query(question)
        count = self._col.count()
        if count >= self._max_size:
            if self._debug:
                print(f"[cache] full ({count}/{self._max_size}) — evicting oldest entry")
            all_items = self._col.get(include=["metadatas"])
            oldest_id = min(
                zip(all_items["ids"], all_items["metadatas"]),
                key=lambda x: x[1].get("timestamp", 0.0),
            )[0]
            self._col.delete(ids=[oldest_id])
        self._col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[{"answer": answer, "timestamp": time.time(), "question": question[:500]}],
        )
        if self._debug:
            print(f"[cache] stored (size: {self._col.count()}/{self._max_size})")

class EfficientRetrievalAgent:
    def __init__(self, hybrid: HybridRetriever, wikifiles_dir: str, top_k: int = NUM_RETRIEVED,
                 max_iterations: int = MAX_ITERATIONS, debug: bool = True,
                 plan_model: Optional[str] = None, answer_model: Optional[str] = None,
                 retriever_model: Optional[str] = None, router_model: str = ROUTER_MODEL,
                 direct_model: Optional[str] = None, fallback_model: str = FALLBACK_MODEL,
                 use_cache: bool = True, use_router: bool = True,
                 cache_dir: str = CACHE_DIR, cache_max_size: int = CACHE_MAX_SIZE,
                 cache_similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        # Per-role models (each falls back to the single default).
        self._plan_model = plan_model or LLM_MODEL
        self._answer_model = answer_model or LLM_MODEL
        self._retriever_model = retriever_model or LLM_MODEL
        self._router_model = router_model
        self._direct_model = direct_model or LLM_MODEL
        self._fallback_model = fallback_model

        self._plan_llm = ChatOpenAI(model=self._plan_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._answer_llm = ChatOpenAI(model=self._answer_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._router_llm = ChatOpenAI(model=self._router_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._direct_llm = ChatOpenAI(model=self._direct_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._fallback_plan_llm = ChatOpenAI(model=self._fallback_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self._fallback_answer_llm = ChatOpenAI(model=self._fallback_model, api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)

        # The hybrid is injected (built once, reused). It makes no chat-LLM calls, so
        # retriever_model does not affect the token tally; we keep it for API parity.
        self._hybrid = hybrid
        self._wikifiles_dir = wikifiles_dir
        self._top_k = top_k
        self._max_iterations = max_iterations
        self._debug = debug
        self._token_usage: dict[str, dict[str, int]] = {}
        self._use_router = use_router
        self._cache_similarity_threshold = cache_similarity_threshold
        self._cache = _SemanticCache(cache_dir=cache_dir, max_size=cache_max_size, debug=debug) if use_cache else None

        # Two graphs: the regular agent (small models) and a fallback agent (strong model).
        self._graph = self._build_graph(self._plan_llm, self._plan_model, self._answer_llm, self._answer_model)
        self._fallback_graph = self._build_graph(
            self._fallback_plan_llm, self._fallback_model, self._fallback_answer_llm, self._fallback_model,
        )

    def get_model_name(self) -> str:
        return f"plan={self._plan_model}, answer={self._answer_model}, router={self._router_model}, direct={self._direct_model}, fallback={self._fallback_model}"

    # ── token-usage tracking (kept from Lab 6.2) ────────────────────
    def _track_usage(self, response, model_name: str) -> None:
        """Accumulate this response's token counts under `model_name`.

        Every LangChain chat response exposes `usage_metadata`, a dict with `input_tokens`,
        `output_tokens`, and `input_token_details` (which holds `cache_read` when the provider
        served part of the prompt from cache). Some responses/models omit it, so we no-op then.
        """
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        bucket = self._token_usage.setdefault(model_name, {"input": 0, "output": 0, "cached": 0})
        bucket["input"] += usage.get("input_tokens", 0)
        bucket["output"] += usage.get("output_tokens", 0)
        details = usage.get("input_token_details", {})
        bucket["cached"] += details.get("cache_read", 0)

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        """Return accumulated token counts keyed by model name (a safe copy)."""
        return {model: dict(counts) for model, counts in self._token_usage.items()}

    def print_token_usage(self) -> None:
        for model, counts in self._token_usage.items():
            print(f"{model}: {counts['input']} input, {counts['output']} output, {counts['cached']} cached")

    def reset_token_usage(self) -> None:
        self._token_usage.clear()

    # ── JSON parsing helper ──────────────────────────────────────────
    def _parse_json_response(self, response) -> dict:
        raw = (response.content if hasattr(response, "content") else str(response)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}

    # ── Step 2: routing + direct answering ───────────────────────────
    def _route_query(self, question: str) -> str:
        """Ask the cheap router model whether this question needs the full agent.

        Returns "direct" (a single retrieval + one small-model answer is enough) or "agent"
        (the full multi-step pipeline). We default to "agent" on any parse failure — the safe,
        slightly-more-expensive path — rather than risk a bad cheap answer.
        """
        response = self._router_llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=f"Question: {question}"),
        ])
        self._track_usage(response, self._router_model)
        route = self._parse_json_response(response).get("route", "agent")
        if self._debug:
            print(f"[agent] router -> {route!r}")
        return route

    def _validate_cached_answer(self, question: str, cached_answer: str) -> bool:
        """On each cache hit, confirm the question is self-contained, not context-dependent, and
        the cached answer actually answers it before trusting the cache. 
        
        This validation could be optimized to run only the first time an entry is hit, but it is kept
        explicit here for instructional clarity.
        """
        response = self._router_llm.invoke([
            SystemMessage(content=_CACHE_VALIDATE_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nCached answer: {cached_answer}"),
        ])
        self._track_usage(response, self._router_model)
        result = self._parse_json_response(response)
        valid = bool(result.get("valid", False))
        if self._debug:
            print(f"[agent] cache validation -> {'VALID' if valid else 'INVALID'}")
        return valid

    def _check_answer_satisfactory(self, question: str, answer: str, context: str = "") -> bool:
        """Decide whether a cheap direct answer is good enough to return (else escalate).

        The retrieved WikiFiles are passed in so the judge can check COMPLETENESS — an answer
        that names only some of the relevant entities the retrieved WikiFiles contain must be
        rejected, not just an answer that is vague or off-topic.
        """
        content = f"Question: {question}\n\nAnswer: {answer}"
        if context:
            content += f"\n\nRetrieved WikiFiles the answer was based on:\n{context}"
        response = self._router_llm.invoke([
            SystemMessage(content=_SATISFACTION_SYSTEM),
            HumanMessage(content=content),
        ])
        self._track_usage(response, self._router_model)
        satisfactory = bool(self._parse_json_response(response).get("satisfactory", False))
        if self._debug:
            print(f"[agent] satisfaction -> {'OK' if satisfactory else 'ESCALATE'}")
        return satisfactory

    def _direct_retrieve_and_answer(self, question: str, conversation_history: list[dict]) -> tuple[str, bool]:
        """Answer via one hybrid retrieval + a single direct LLM call. Returns (answer, ok)."""
        results = self._hybrid.getTopK(question, self._top_k)
        context = "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)
        user_content = ""
        if conversation_history:
            conv_lines = [f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history[-6:]]
            user_content += f"Prior conversation:\n{chr(10).join(conv_lines)}\n\n"
        user_content += f"Retrieved WikiFiles:\n{context}\n\nQuestion: {question}"
        response = self._direct_llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=user_content),
        ])
        self._track_usage(response, self._direct_model)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, self._check_answer_satisfactory(question, answer, context)

    # — tool: retrieve all WikiFiles on a date / short range (filename-encoded dates) —
    def retrieve_by_date(self, start_date: str, end_date: Optional[str] = None) -> list[tuple[str, str]]:
        """Return (filename, content) for every e-mail on start_date, or within the inclusive
        range [start_date, end_date] — BOTH endpoints included. (From Lab 5.2.)"""
        start = dateutil_parser.parse(start_date).date()
        end = dateutil_parser.parse(end_date).date() if end_date else start
        results = []
        for path in glob.glob(os.path.join(self._wikifiles_dir, "mail_*.txt")):
            m = _FILENAME_RE.match(os.path.basename(path))
            if not m:
                continue
            wikifile_date = date(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
            if start <= wikifile_date <= end:
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        results.append((os.path.basename(path), f.read()))
                except OSError:
                    pass
        if self._debug:
            print(f"[agent] by_date {start}..{end}: {len(results)} e-mail(s)")
        return results

    def _build_graph(self, plan_llm, plan_model: str, answer_llm, answer_model: str):
        def retrieve_node(state: _AgentState) -> dict:
            wikifile_bodies = dict(state["wikifile_bodies"])
            executed = list(state["executed_queries"])
            for query in state["pending_queries"]:
                if self._debug:
                    print(f"[agent] retrieving for: {query!r}")
                for name, content, _ in self._hybrid.getTopK(query, self._top_k):
                    wikifile_bodies[name] = content
                executed.append(query)
            return {"wikifile_bodies": wikifile_bodies, "executed_queries": executed,
                    "pending_queries": [], "iterations": state["iterations"] + 1}

        def retrieve_by_date_node(state: _AgentState) -> dict:
            dr = state.get("pending_date_range", {})
            start, end = dr.get("start", ""), (dr.get("end", "") or None)
            if not start:
                return {"pending_date_range": {}, "iterations": state["iterations"] + 1}
            wikifile_bodies = dict(state["wikifile_bodies"])
            for name, content in self.retrieve_by_date(start, end):
                wikifile_bodies[name] = content
            tag = f"DATE:{start}" + (f"..{end}" if end else "")
            return {"wikifile_bodies": wikifile_bodies,
                    "executed_queries": list(state["executed_queries"]) + [tag],
                    "pending_date_range": {}, "iterations": state["iterations"] + 1}

        def plan_node(state: _AgentState) -> dict:
            """The agent's brain: pick the next action as a JSON object (planner LLM)."""
            if state["iterations"] >= self._max_iterations:
                if self._debug:
                    print("[agent] max iterations — forcing answer.")
                return {"next_action": "answer", "pending_queries": [], "pending_date_range": {}, "clarification_question": ""}

            wikifile_summary = "\n\n---\n\n".join(
                f"[{name}]\n{content}" for name, content in list(state["wikifile_bodies"].items())[:20]
            )
            executed_str = "\n".join(f"- {q}" for q in state["executed_queries"]) or "(none)"
            clarif_str = "\n".join(state["clarification_history"]) or "(none)"
            conv_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                                 for m in state["conversation_history"][-6:]) or "(none)"
            user_content = (
                f"Prior conversation:\n{conv_str}\n\n"
                f"Interactions so far:\n{clarif_str}\n\n"
                f"Queries already executed:\n{executed_str}\n\n"
                f"WikiFiles retrieved ({len(state['wikifile_bodies'])} total):\n\n{wikifile_summary}"
            )
            response = plan_llm.invoke([SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user_content)])
            self._track_usage(response, plan_model)
            result = self._parse_json_response(response)
            action = result.get("action", "answer")
            if self._debug:
                print(f"[agent] action={action} reasoning={result.get('reasoning', '')!r}")
            if action == "retrieve":
                return {"next_action": "retrieve", "pending_queries": result.get("queries", []),
                        "pending_date_range": {}, "clarification_question": ""}
            if action == "by_date":
                return {"next_action": "by_date", "pending_queries": [],
                        "pending_date_range": {"start": result.get("start_date", ""), "end": result.get("end_date", "")},
                        "clarification_question": ""}
            if action == "clarify":
                return {"next_action": "clarify", "pending_queries": [], "pending_date_range": {},
                        "clarification_question": result.get("clarification", "Could you clarify your question?")}
            return {"next_action": "answer", "pending_queries": [], "pending_date_range": {}, "clarification_question": ""}

        def clarify_node(state: _AgentState) -> dict:
            # In batch mode (query()) there is no user to answer, so we record that the
            # clarification was unavailable; the clarify->plan edge then lets the agent
            # re-plan with what it has instead of blocking on input().
            question = state["clarification_question"]
            history = list(state["clarification_history"])
            if state["mode"] == "chat":
                print(f"\nAssistant: {question}")
                history.append(f"Q: {question}\nA: {input('You: ').strip()}")
            else:  # batch mode never blocks on input
                history.append(f"Q: {question}\n[no clarification available in batch mode]")
            return {"clarification_history": history, "clarification_question": ""}

        def answer_node(state: _AgentState) -> dict:
            wikifile_context = "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c in state["wikifile_bodies"].items())
            conv_str = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in state["conversation_history"][-6:])
            clarif_str = "\n".join(state["clarification_history"])
            parts = []
            if conv_str:
                parts.append(f"Prior conversation:\n{conv_str}")
            if clarif_str:
                parts.append(f"Interactions so far:\n{clarif_str}")
            parts.append(f"Retrieved WikiFiles:\n{wikifile_context}")
            response = answer_llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content="\n\n".join(parts))])
            self._track_usage(response, answer_model)
            return {"answer": response.content if hasattr(response, "content") else str(response), "done": True}

        def route_from_plan(state: _AgentState) -> str:
            action = state.get("next_action", "answer")
            if action == "retrieve" and state["pending_queries"]:
                return "retrieve"
            if action == "by_date" and state.get("pending_date_range", {}).get("start"):
                return "retrieve_by_date"
            if action == "clarify":
                return "clarify"
            return "answer"

        graph = StateGraph(_AgentState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("retrieve_by_date", retrieve_by_date_node)
        graph.add_node("plan", plan_node)
        graph.add_node("clarify", clarify_node)
        graph.add_node("answer", answer_node)
        graph.set_entry_point("plan")
        graph.add_edge("retrieve", "plan")
        graph.add_edge("retrieve_by_date", "plan")
        graph.add_conditional_edges("plan", route_from_plan, {
            "retrieve": "retrieve", "retrieve_by_date": "retrieve_by_date",
            "clarify": "clarify", "answer": "answer",
        })
        graph.add_edge("clarify", "plan")
        graph.add_edge("answer", END)
        return graph.compile()

    def _run_agent(self, question: str, mode: str, conversation_history: Optional[list[dict]],
                   use_fallback: bool = False) -> str:
        graph = self._fallback_graph if use_fallback else self._graph
        initial: _AgentState = {
            "conversation_history": conversation_history or [],
            "clarification_history": [question],
            "pending_queries": [question],
            "pending_date_range": {},
            "executed_queries": [],
            "wikifile_bodies": {},
            "iterations": 0,
            "mode": mode,
            "next_action": "plan",
            "clarification_question": "",
            "answer": "",
            "done": False,
        }
        return graph.invoke(initial)["answer"]

    def _run(self, question: str, mode: str, conversation_history: Optional[list[dict]] = None) -> str:
        """Cache -> route -> (direct | full agent). Storing every answer back into the cache."""
        # 1. Semantic cache (Step 1). A validated hit is by far the cheapest outcome.
        if self._cache:
            hit = self._cache.lookup(question, self._cache_similarity_threshold)
            if hit:
                cached_answer, _ = hit
                if self._validate_cached_answer(question, cached_answer):
                    if self._debug:
                        print("[agent] returning validated cached answer")
                    return cached_answer

        # 2. Route (Step 2). With routing disabled we force the full agent — this reproduces
        #    the un-optimized Lab 6.2 agent, which the measure step compares against.
        route = self._route_query(question) if self._use_router else "agent"

        # 3. Direct path: one retrieval + one small-model answer; escalate if unsatisfactory.
        if route == "direct":
            answer, ok = self._direct_retrieve_and_answer(question, conversation_history or [])
            if not ok:
                if self._debug:
                    print("[agent] direct answer unsatisfactory — escalating to full agent")
                answer = self._run_agent(question, mode, conversation_history, use_fallback=True)
            if self._cache:
                self._cache.store(question, answer)
            return answer

        # 4. Full agent path.
        answer = self._run_agent(question, mode, conversation_history, use_fallback=False)
        if self._cache:
            self._cache.store(question, answer)
        return answer

    def query(self, question: str) -> str:
        """Batch mode — no clarification prompts (for the automated cost suite)."""
        return self._run(question, "batch")

    def chat(self) -> None:
        print("Efficient research agent (cache + router). Type your question; 'exit'/'quit' to stop.\n")
        conversation_history: list[dict] = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue
            try:
                answer = self._run(user_input, "chat", conversation_history)
            except Exception as e:
                print(f"Error: {e}")
                continue
            print(f"\nAssistant: {answer}\n")
            if answer.strip() == "Exiting":
                print("Goodbye.")
                break
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})


# ─── Step 3: measure — cost harness (kept from Lab 6.2) ───────────────
def estimate_cost_usd(usage: dict[str, dict[str, int]]) -> float:
    """Rough USD estimate from a token tally using APPROX_PRICE_PER_MTOK.

    Naive: input * in_rate + output * out_rate. Cached tokens are counted at the full input
    rate here (real providers discount them), so this is an UPPER-ish estimate. Unknown models
    contribute 0 — add them to APPROX_PRICE_PER_MTOK to include them.
    """
    total = 0.0
    for model, counts in usage.items():
        price = APPROX_PRICE_PER_MTOK.get(model)
        if not price:
            continue
        total += counts["input"] / 1_000_000 * price["input"]
        total += counts["output"] / 1_000_000 * price["output"]
    return total


def run_experiment(questions: list[str], *, use_cache: bool, use_router: bool,
                   wikifiles_dir: str = WIKIFILES_DIR_DEFAULT, label: Optional[str] = None) -> dict[str, dict[str, int]]:
    """Run `query()` over a small question set with the given optimizations and report per-model
    token usage (+ a rough USD estimate). Returns the token tally.

    Flags:
      - use_cache=False, use_router=False -> the un-optimized Lab 6.2 agent (baseline).
      - use_cache=True,  use_router=True  -> the optimized agent (cache + router).
    """
    heading = label or f"cache={use_cache} router={use_router}"
    print("\n" + "=" * 78)
    print(f"EXPERIMENT: {heading}")
    print("=" * 78)

    db = build_or_load_db(wikifiles_dir, CHROMA_DIR)
    hybrid = HybridRetriever(wikifiles_dir=wikifiles_dir, db=db)
    agent = EfficientRetrievalAgent(hybrid=hybrid, wikifiles_dir=wikifiles_dir, debug=False,
                                    use_cache=use_cache, use_router=use_router)
    agent.reset_token_usage()

    for i, q in enumerate(questions, 1):
        answer = agent.query(q)
        one_line = " ".join(answer.split())
        preview = one_line if len(one_line) <= 160 else one_line[:157] + "..."
        print(f"\nQ{i}. {q}\n    -> {preview}")

    print("\nToken usage:")
    agent.print_token_usage()
    usage = agent.get_token_usage()
    print(f"Approx. cost (USD, see disclaimer): ${estimate_cost_usd(usage):.4f}")
    return usage


def _safe_experiment(*args, **kwargs) -> None:
    """Run one experiment; if the provider errors, skip it and keep going. Models on OpenRouter
    can be rate-limited upstream (HTTP 429) or transiently 5xx, so one bad run must not crash the
    whole measure step.
    Catch-all: any model failure is non-fatal; that's the point of the harness."""
    label = kwargs.get("label") or (args[1] if len(args) > 1 else "?")
    try:
        run_experiment(*args, **kwargs)
    except Exception as e:
        print(f"[skipped] {label}: {type(e).__name__} — {e}")
        print(
            "          The model may be rate-limited upstream; retry later or confirm that your "
            "program-provided OpenRouter API key is set correctly. "
            "OpenRouter key: https://openrouter.ai/settings/integrations"
        )


def main() -> None:
    require_api_key()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wikifiles_dir = args[0] if args else WIKIFILES_DIR_DEFAULT

    # Build the DB once up front (both experiments reuse the same persisted vectors).
    build_or_load_db(wikifiles_dir, CHROMA_DIR)

    print("\n### Step 3 — measure: un-optimized baseline vs optimized (cache + router) ###")
    _safe_experiment(EXPERIMENT_QUESTIONS, use_cache=False, use_router=False, wikifiles_dir=wikifiles_dir,
                     label="BASELINE — un-optimized full agent (no cache, no router)")
    _safe_experiment(EXPERIMENT_QUESTIONS, use_cache=True, use_router=True, wikifiles_dir=wikifiles_dir,
                     label="OPTIMIZED — semantic cache + query router")

    print("\nCompare the two 'Approx. cost' lines above — AND compare the ANSWERS question by")
    print("question: cheaper is only a win if the optimized answers stay equivalent to the")
    print("baseline (see the 'IMPORTANT' note in this file's docstring).")
    print("For a stronger cache validation, re-run this script WITHOUT deleting semantic_cache/:")
    print("the repeated questions should then hit the cache and the second run's cost should drop.")


if __name__ == "__main__":
    main()
