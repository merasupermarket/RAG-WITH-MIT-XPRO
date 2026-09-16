"""Lab 4.2 (SOLUTION) — graph-augmented retrieval over the WikiFiles database.

Developer: Gaurav Singh, 09/12/2026
git- https://github.com/merasupermarket/RAG-WITH-MIT-XPRO

Run:  python lab_4_2_graph_retrieval_solution.py "annotatedWikiFiles.json"

Some answers depend on RELATIONSHIPS between Wikifiles — the rest of a conversation
thread, or other WikiFiles on the same topic, that a keyword or vector search
won't surface on their own. This lab builds a knowledge graph of the WikiFiles
(NetworkX) and uses it to enrich retrieval:

  hybrid search finds 5 seed WikiFiles
    -> for each seed, follow graph edges to add its topic neighbors
       and the earliest WikiFiles on each of its TOPICS
    -> deduplicate, then give the LLM a context that clearly separates the
       main retrieved WikiFiles from the topic context.

The graph is built from annotatedWikiFiles.json, which the faculty provides. Each
record already carries sender/recipients/mentions/threadID/topics (extracting
those with an LLM over thousands of WikiFiles is slow and is done for you), plus
the raw text — so this whole lab runs from that one file.

Setup
-----
1. Create the environment (one-time). Either use conda:
       conda env create -f environment.yml
       conda activate ragcourse
   or a plain virtual environment + pip:
       python -m venv .venv
       #  Windows:      .venv\Scripts\activate
       #  macOS/Linux:  source .venv/bin/activate
       pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25 networkx
2. Add the OpenRouter API key provided for this program. Create a file
   named ".env" in this folder containing a single line:
       OPENROUTER_API_KEY=sk-or-your-key-here
   (or set it in your shell —  Windows:  setx OPENROUTER_API_KEY sk-or-...
    macOS/Linux:  export OPENROUTER_API_KEY=sk-or-...)
3. Data: annotatedWikiFiles.json (provided by the faculty) in this folder, or pass
   its path as the first argument. The vector DB is persisted to ./chroma_db
   (delete it to rebuild); the graph is rebuilt from the JSON each run (it's fast).

Sample questions to try (over the WikiFiles corpus)
-----------------------------------------------
Questions that need a whole conversation or the history of a topic:
    "How do you define Apollo 11 as success or failure?"
        ->  The provided WikiFiles do not contain information that defines Apollo 11 as a success or a failure.
    "Hightlight lows and highs of the Apollo 11 mission."
        -> The WikiFiles you provided describe many **high points** associated with Apollo 11—its historic Moon landing, the astronauts receiving a Congressional Gold Medal, the extensive anniversary celebrations (photo releases, restored audio and video, commemorative coins, museum festivals, films, and even a Google Doodle). A quoted group of British scientists also praised the mission as “technically brilliant” and “the greatest technical achievement of mankind to date.”
            However, the same files do **not** mention any **low points** or difficulties experienced during the mission itself (e.g., technical glitches, near‑misses, or challenges in the descent and landing). Because that information is absent from the provided WikiFiles, I cannot list specific lows.
            **Summary based on the available content**
            - **Highs:** successful Moon landing; historic astronaut honors (Congressional Gold Medal); extensive public commemorations and media releases; recognition as a landmark technical achievement.  
            - **Lows:** not documented in the supplied WikiFiles.    


Sample JSON input (annotatedWikiFiles.json)
-----------------------------------------------
{
  "WikiFiles": [
    {
      "filename": "Apollo_11.txt",
      "rawText": "The Apollo 11 mission was a historic event that marked the first successful manned landing on the Moon. The astronauts received a Congressional Gold Medal for their achievement.",
      "topics": ["Apollo 11", "Moon Landing"],
      "subject": "Apollo 11 moon mission",
      "seq_num": 1
    },
    {
      "filename": "Apollo_11_Celebrations.txt",
      "rawText": "The Apollo 11 mission was celebrated with various events, including the release of photos and videos, commemorative coins, and museum exhibitions.",
      "topics": ["Apollo 11", "Celebrations"],
      "subject": "Apollo 11 moon mission",
      "seq_num": 2
    }]
}           
"""
import json
import os
import re
import sys
from abc import ABC, abstractmethod

import networkx as nx
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
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
HYBRID_FETCH_K = 5    # seed WikiFiles from hybrid search
THREAD_NEIGHBORS = 5  # closest WikiFiles in a seed's thread
TOPIC_EARLIEST = 5    # earliest WikiFiles per topic

SYSTEM_PROMPT = """You are a helpful assistant for WikiFiles \
You answer questions by drawing information exclusively from the company WikiFiles \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided WikiFiles, answer clearly and concisely.
- If the provided WikiFiles do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided WikiFiles."""

GRAPH_SYSTEM_PROMPT = SYSTEM_PROMPT + """

The context is organized into three kinds of WikiFiles:
- RETRIEVED Wiki Files: the main WikiFiles matched to the question.
- THREAD CONTEXT: other WikiFiles from the same conversation, for continuity.
- TOPIC CONTEXT: earlier WikiFiles on the same topic, to show how it developed.
Use the topic context to understand relationships, but base your
answer on what the WikiFiles actually say."""


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


def chat_loop(response):
    print("Chat over the Wikifiles DB. Type your question; 'exit'/'quit' to stop.\n")
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

# ─── Data loading + hybrid stack (built from annotatedWikiFiles.json) ───
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


def _seq_num(filename: str) -> int:
    """Trailing number in a filename like mail_01_01_14_225.txt — used to order
    WikiFiles within a thread and within a topic."""
    m = re.search(r"_(\d+)\.txt$", filename)
    return int(m.group(1)) if m else 0


def load_annotated(path: str) -> tuple[list[dict], dict[str, str]]:
    """Load the faculty annotatedWikiFiles.json.

    Each JSON record becomes its own unique document so repeated filenames do not
    collapse into a single body. This keeps the graph and retrieval pipeline
    working with the topic-only annotated dataset you currently have.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    WikiFiles = []
    body_by_filename: dict[str, str] = {}
    for idx, e in enumerate(data["WikiFiles"]):
        record = dict(e)
        record["record_id"] = f"{e['filename']}::record_{idx}"
        WikiFiles.append(record)
        body_by_filename[record["record_id"]] = e.get("rawText", "")
    return WikiFiles, body_by_filename


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(WikiFiles: list[dict], chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    print(f"Building vector DB (first run — embedding {len(WikiFiles)} WikiFiles)...")
    docs = [
        Document(
            page_content=e.get("rawText", ""),
            metadata={"source": e.get("record_id", e["filename"])}
        )
        for e in WikiFiles
    ]
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


class BaseRetriever(ABC):
    def __init__(self, llm_model: str = LLM_MODEL, system_prompt: str = SYSTEM_PROMPT):
        self._llm = ChatOpenAI(
            model=llm_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
        self._system_prompt = system_prompt
        self._history = [SystemMessage(content=system_prompt)]

    @abstractmethod
    def retrievedContext(self, query: str) -> str: ...

    def _build_user_message(self, query: str, context: str) -> str:
        return f"Context (WikiFiles):\n{context}\n\nQuestion: {query}"

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

class HybridRetriever:
    """Score-fusion of BM25 + vector, returning (filename, content, score)."""

    def __init__(self, WikiFiles: list[dict], db: Chroma):
        self._paths = [e.get("record_id", e["filename"]) for e in WikiFiles]
        self._contents = [e.get("rawText", "") for e in WikiFiles]
        self._bm25 = BM25Okapi([tokenize(c) for c in self._contents])
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

# ─── Build the knowledge graph (provided infrastructure) ─────────────
def build_graph(wikis: list[dict]) -> nx.DiGraph:
    """Build a topic-only knowledge graph for the available annotated records.

    This dataset contains per-record topics but does not include thread / sender /
    mention metadata, so the graph intentionally only models each record -> topic
    relationships and uses those topics for graph expansion.
    """
    G = nx.DiGraph()
    for idx, e in enumerate(wikis):
        fname = e.get("filename")
        if not fname:
            continue

        record_id = e.get("record_id", f"{fname}::{idx}")
        wiki_node = f"wikis:{record_id}"
        G.add_node(
            wiki_node,
            node_type="wiki",
            filename=fname,
            record_id=record_id,
            seq_num=e.get("seq_num", idx),
        )

        for topic in e.get("topics") or []:
            topic_node = f"topic:{topic}"
            if not G.has_node(topic_node):
                G.add_node(topic_node, node_type="topic", topic=topic)
            G.add_edge(wiki_node, topic_node, edge_type="relates_to")
    return G


# ─── Graph-augmented retriever ───────────────────────────────────────
class GraphRetriever(BaseRetriever):
    def __init__(self, graph: nx.DiGraph, hybrid: HybridRetriever, body_by_filename: dict[str, str], **kwargs):
        super().__init__(system_prompt=GRAPH_SYSTEM_PROMPT, **kwargs)
        self._G = graph
        self._hybrid = hybrid
        self._bodies = body_by_filename
        # Pre-index thread members and topic members, each sorted by seq_num.
        self._thread_wikis: dict[str, list[str]] = {}
        self._topic_wikis: dict[str, list[str]] = {}
        for node, data in graph.nodes(data=True):
            if data.get("node_type") != "wiki":
                continue
            for _, target, edata in graph.out_edges(node, data=True):
                if edata.get("edge_type") == "belongs_to":
                    self._thread_wikis.setdefault(target, []).append(node)
                elif edata.get("edge_type") == "relates_to":
                    self._topic_wikis.setdefault(target, []).append(node)
        for members in self._thread_wikis.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))
        for members in self._topic_wikis.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))

    # — graph traversal helpers (provided) —
    def _filename(self, wiki_node: str) -> str:
        return self._G.nodes[wiki_node].get("record_id", self._G.nodes[wiki_node].get(
            "filename", wiki_node.removeprefix("wikis:")
        ))

    def _thread_for_wiki(self, wiki_node: str):
        # This dataset does not contain thread metadata, so thread neighbors are not
        # available. Keep the method as a no-op for compatibility with the graph
        # traversal code.
        return None

    def _closest_thread_neighbors(self, wiki_node: str) -> list[str]:
        """No thread expansion is available when the dataset lacks thread metadata."""
        return []

    def _topics_for_wiki(self, wiki_node: str) -> list[str]:
        return [t for _, t, ed in self._G.out_edges(wiki_node, data=True) if ed.get("edge_type") == "relates_to"]

    def _earliest_topic_wikis(self, topic_node: str) -> list[str]:
        return [self._filename(n) for n in self._topic_wikis.get(topic_node, [])[:TOPIC_EARLIEST]]

    def _body(self, filename: str) -> str:
        return self._bodies.get(filename, f"[content not found: {filename}]")

    def _assemble_context(self, seeds, thread_neighbors, topic_context) -> str:
        sections = []
        for fname in seeds:
            sections.append(f"=== RETRIEVED WIKI: {fname} ===\n{self._body(fname)}")
            for nb in thread_neighbors.get(fname, []):
                sections.append(f"--- THREAD CONTEXT for {fname}: {nb} ---\n{self._body(nb)}")
            for topic, files in topic_context.get(fname, {}).items():
                for ef in files:
                    sections.append(f'--- TOPIC CONTEXT "{topic}" (via {fname}): {ef} ---\n{self._body(ef)}')
        return ("\n\n" + "=" * 70 + "\n\n").join(sections)

    def retrievedContext(self, query: str) -> str:
        """Graph-augmented retrieval: hybrid seeds + thread + topic neighbours, deduped.

        Get HYBRID_FETCH_K seed wikis, then for each seed, walk the graph to add its
        thread neighbors and the earliest wikis on its topics. A `union` dict keyed
        by filename keeps each wiki's single source label at its highest priority
        (seed > thread > topic), and we drop neighbors that are themselves seeds."""
        seeds = [name for name, _, _ in self._hybrid.getTopK(query, HYBRID_FETCH_K)]
        union: dict[str, str] = {}
        thread_neighbors: dict[str, list[str]] = {}
        topic_context: dict[str, dict[str, list[str]]] = {}

        for fname in seeds:
            union[fname] = "seed"  # seeds always win the label
            wiki_node = f"wiki:{fname}"
            if not self._G.has_node(wiki_node):
                continue
            nbrs = [n for n in self._closest_thread_neighbors(wiki_node) if n != fname]
            thread_neighbors[fname] = nbrs
            for nb in nbrs:
                union.setdefault(nb, "thread")
            tctx: dict[str, list[str]] = {}
            for topic_node in self._topics_for_wiki(wiki_node):
                earliest = [ef for ef in self._earliest_topic_wikis(topic_node) if ef != fname]
                if earliest:
                    tctx[topic_node.removeprefix("topic:")] = earliest
                    for ef in earliest:
                        union.setdefault(ef, "topic")
            topic_context[fname] = tctx

        # Drop thread/topic neighbors that are themselves seeds (avoid redundancy).
        for fname in seeds:
            thread_neighbors[fname] = [n for n in thread_neighbors.get(fname, []) if union.get(n) != "seed"]
            topic_context[fname] = {
                t: [ef for ef in files if union.get(ef) != "seed"]
                for t, files in topic_context.get(fname, {}).items()
            }
        return self._assemble_context(seeds, thread_neighbors, topic_context)


def main():
    require_api_key()
    annotated_path = sys.argv[1] if len(sys.argv) > 1 else "annotatedWikiFiles.json"
    print(f"Loading {annotated_path} ...")
    wikiFiles, body_by_filename = load_annotated(annotated_path)
    print(f"  {len(wikiFiles)} wiki files loaded.")
    db = build_or_load_db(wikiFiles, CHROMA_DIR)
    hybrid = HybridRetriever(wikiFiles, db)
    print("Building knowledge graph ...")
    graph = build_graph(wikiFiles)
    print(f"  graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    GraphRetriever(graph=graph, hybrid=hybrid, body_by_filename=body_by_filename).chat()


if __name__ == "__main__":
    main()
