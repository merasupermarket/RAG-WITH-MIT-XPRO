"""lab_4_3_retrieval_Graph_Diagnosis.py — evaluate the Graph retriever with RAGAS.

Developer: Gaurav Singh, 09/14/2026
git- https://github.com/merasupermarket/RAG-WITH-MIT-XPRO

Setup
-----
1. Ensure that you have completed the program's one-time environment and
   OpenRouter API key setup, and activate the configured environment.

2. Install the required dependencies, if they are not already available:
       pip install -r requirements.txt
       pip install networkx

3. Locate the Wikifiles data: Ensure that the provided .txt Wikifiles are
   available in the 'WikiFiles' folder, or pass the folder path when
   running the script. The evaluation questions are provided in
   testInputs.json.

Each run answers every test question with the chosen retriever. Then an LLM judge
scores each answer pass/fail against grading notes (a RAGAS DiscreteMetric) and
writes a CSV under ragas_experiments/experiments/. 

- Step 1: Complete the evaluation experiment so each test question is answered
   by the selected retriever and scored by the LLM judge.
 
- Step 2: Refine the evaluation metric, re-run the experiment, 
   and compare how the scoring results change.
   
Sample questions to try
-----------------------
testInputs.json holds the evaluation questions, each with grading_notes the judge
checks the answer against. Run the evaluation separately for BM25, vector, and hybrid retrieval, 
then compare the pass rates across the three approaches:
    python lab_3_2_retrieval_Baseline Diagnosis_solution.py bm25   WikiFiles
    python lab_3_2_retrieval_Baseline Diagnosis_solution.py vector WikiFiles
    python lab_3_2_retrieval_Baseline Diagnosis_solution.py hybrid WikiFiles

Expected runtime: A full evaluation across BM25, vector, and hybrid retrieval may take approximately 
5 minutes. The first vector/hybrid run may take longer while the corpus is embedded. 
Runtime may vary depending on provider availability, rate limits, and environment.
After the first run, tune the metric (Step 2) and re-run to see how the pass/fail counts change. 

"""
import argparse
import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import sys
from abc import ABC, abstractmethod

import networkx as nx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from rank_bm25 import BM25Okapi

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
EMBEDDING_MODEL = "openai/text-embedding-3-small"
JUDGE_MODEL = "openai/gpt-5.4-mini"  # judge model; try a different one in Step 2
CHROMA_DIR = "chroma_db"
NUM_RETRIEVED = 4
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5
HYBRID_FETCH_K = 5    # seed WikiFiles from hybrid search
THREAD_NEIGHBORS = 5  # closest WikiFiles in a seed's thread
TOPIC_EARLIEST = 5    # earliest WikiFiles per topic

SYSTEM_PROMPT = """You are a helpful assistant for WikiFiles. \
You answer questions by drawing information exclusively from the Wikipedia files refered as WikiFiles \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided WikiFiles, answer clearly and concisely.
- If the provided WikiFiles do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Do not answer questions that are unrelated to the content of the provided WikiFiles."""

GRAPH_SYSTEM_PROMPT = SYSTEM_PROMPT + """

The context is organized into three kinds of WikiFiles:
- RETRIEVED E-MAIL: the main WikiFiles matched to the question.
- THREAD CONTEXT: other WikiFiles from the same conversation, for continuity.
- TOPIC CONTEXT: earlier WikiFiles on the same topic, to show how it developed.
Use the topic context to understand relationships, but base your
answer on what the WikiFiles actually say."""


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
    """Load the faculty annotatedWikiFiles.json. Returns (WikiFiles, body_by_filename).

    The annotated JSON contains multiple entries per filename. Preserve all of the
    raw text for each filename instead of overwriting earlier chunks with the last
    one, otherwise graph retrieval ends up with only a partial body for that file.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    WikiFiles = data["WikiFiles"]
    body_by_filename: dict[str, str] = {}
    for e in WikiFiles:
        filename = e["filename"]
        raw_text = e.get("rawText", "")
        if not raw_text:
            continue
        prev = body_by_filename.get(filename, "")
        if prev:
            body_by_filename[filename] = f"{prev}\n\n---\n\n{raw_text}"
        else:
            body_by_filename[filename] = raw_text
    return WikiFiles, body_by_filename

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,  # OpenRouter needs raw text, not pre-tokenized input
    )


def build_or_load_db(WikiFiles: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings())
    docs = []
    for fname in sorted(os.listdir(WikiFiles)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(WikiFiles, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
    return Chroma.from_documents(docs, get_embeddings(), persist_directory=chroma_dir)


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


# ─── Build the knowledge graph (provided infrastructure) ─────────────
def build_graph(WikiFiles: list[dict]) -> nx.DiGraph:
    """Build a directed graph with email / thread / person / topic nodes and the
    five edge types: belongs_to (email->thread), sent (person->email),
    received_by (email->person), mentions (email->person), relates_to (email->topic)."""
    G = nx.DiGraph()
    for e in WikiFiles:
        fname = e["filename"]
        wiki_node = f"wikiFile:{fname}"
        G.add_node(wiki_node, node_type="wiki", filename=fname, seq_num=_seq_num(fname))

        for topic in e.get("topics") or []:
            topic_node = f"topic:{topic}"
            if not G.has_node(topic_node):
                G.add_node(topic_node, node_type="topic", topic=topic)
            G.add_edge(wiki_node, topic_node, edge_type="relates_to")
    return G

# ─── Retriever stack (built in Modules 1-2; provided here) ───────────
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

    def query(self, question: str) -> str:
        context = self.retrievedContext(question)
        messages = [
            SystemMessage(content=self._system_prompt),
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


class Bm25Retriever(BaseRetriever):
    def __init__(self, WikiFiles: str, **kwargs):
        super().__init__(**kwargs)
        self._paths, self._contents = _load_wikifiles(WikiFiles)
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._paths[i], self._contents[i], scores[i]) for i in top]

    def retrievedContext(self, query: str) -> str:
        return "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c, _ in self.getTopK(query, NUM_RETRIEVED))


class VectorRetriever(BaseRetriever):
    def __init__(self, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata.get("source", "unknown"), d.page_content, s) for d, s in results]

    def retrievedContext(self, query: str) -> str:
        return "\n\n---\n\n".join(f"[{n}]\n{c}" for n, c, _ in self.getTopK(query, NUM_RETRIEVED))


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

# ─── Graph-augmented retriever ───────────────────────────────────────
class GraphRetriever(BaseRetriever):
    def __init__(self, graph: nx.DiGraph, hybrid: HybridRetriever, body_by_filename: dict[str, str], **kwargs):
        super().__init__(system_prompt=GRAPH_SYSTEM_PROMPT, **kwargs)
        self._G = graph
        self._hybrid = hybrid
        self._bodies = body_by_filename
        # Pre-index thread members and topic members, each sorted by seq_num.
        self._thread_wikiFiles: dict[str, list[str]] = {}
        self._topic_wikiFiles: dict[str, list[str]] = {}
        for node, data in graph.nodes(data=True):
            if data.get("node_type") != "wiki":
                continue
            for _, target, edata in graph.out_edges(node, data=True):
                if edata.get("edge_type") == "belongs_to":
                    self._thread_wikiFiles.setdefault(target, []).append(node)
                elif edata.get("edge_type") == "relates_to":
                    self._topic_wikiFiles.setdefault(target, []).append(node)
        for members in self._thread_wikiFiles.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))
        for members in self._topic_wikiFiles.values():
            members.sort(key=lambda n: graph.nodes[n].get("seq_num", 0))

    # — graph traversal helpers (provided) —
    def _filename(self, wiki_node: str) -> str:
        return self._G.nodes[wiki_node].get("filename", wiki_node.removeprefix("wikiFile:"))

    def _thread_for_wikifile(self, wiki_node: str):
        for _, target, edata in self._G.out_edges(wiki_node, data=True):
            if edata.get("edge_type") == "belongs_to":
                return target
        return None

    def _closest_thread_neighbors(self, wiki_node: str) -> list[str]:
        """Up to THREAD_NEIGHBORS thread members closest to this e-mail (by seq_num),
        expanding outward from its position and excluding the e-mail itself."""
        thread_node = self._thread_for_wikifile(wiki_node)
        if thread_node is None:
            return []
        members = self._thread_wikiFiles.get(thread_node, [])
        if wiki_node not in members:
            return [self._filename(n) for n in members[:THREAD_NEIGHBORS]]
        idx = members.index(wiki_node)
        out, lo, hi = [], idx - 1, idx + 1
        while len(out) < THREAD_NEIGHBORS and (lo >= 0 or hi < len(members)):
            if lo >= 0:
                out.append(self._filename(members[lo])); lo -= 1
            if len(out) < THREAD_NEIGHBORS and hi < len(members):
                out.append(self._filename(members[hi])); hi += 1
        return out

    def _topics_for_wikifile(self, wiki_node: str) -> list[str]:
        return [t for _, t, ed in self._G.out_edges(wiki_node, data=True) if ed.get("edge_type") == "relates_to"]

    def _earliest_topic_wikiFiles(self, topic_node: str) -> list[str]:
        return [self._filename(n) for n in self._topic_wikiFiles.get(topic_node, [])[:TOPIC_EARLIEST]]

    def _body(self, filename: str) -> str:
        return self._bodies.get(filename, f"[content not found: {filename}]")

    #def _assemble_context(self, seeds, thread_neighbors, topic_context) -> str:
    def _assemble_context(self, seeds, topic_context) -> str:    
        sections = []
        for fname in seeds:
            sections.append(f"=== RETRIEVED E-MAIL: {fname} ===\n{self._body(fname)}")
            # for nb in thread_neighbors.get(fname, []):
            #     sections.append(f"--- THREAD CONTEXT for {fname}: {nb} ---\n{self._body(nb)}")
            for topic, files in topic_context.get(fname, {}).items():
                for ef in files:
                    sections.append(f'--- TOPIC CONTEXT "{topic}" (via {fname}): {ef} ---\n{self._body(ef)}')
        return ("\n\n" + "=" * 70 + "\n\n").join(sections)

    def retrievedContext(self, query: str) -> str:
        """Graph-augmented retrieval: hybrid seeds + topic neighbours, deduped.

        Get HYBRID_FETCH_K seed WikiFiles, then for each seed, walk the graph to add its
        thread neighbors and the earliest WikiFiles on its topics. A `union` dict keyed
        by filename keeps each e-mail's single source label at its highest priority
        (seed > topic), and we drop neighbors that are themselves seeds."""
        seeds = [name for name, _, _ in self._hybrid.getTopK(query, HYBRID_FETCH_K)]
        union: dict[str, str] = {}
        #thread_neighbors: dict[str, list[str]] = {}
        topic_context: dict[str, dict[str, list[str]]] = {}

        for fname in seeds:
            union[fname] = "seed"  # seeds always win the label
            wiki_node = f"wikiFile:{fname}"
            if not self._G.has_node(wiki_node):
                continue
            nbrs = [n for n in self._closest_thread_neighbors(wiki_node) if n != fname]
            #thread_neighbors[fname] = nbrs
            #for nb in nbrs:
            #    union.setdefault(nb, "thread")
            tctx: dict[str, list[str]] = {}
            for topic_node in self._topics_for_wikifile(wiki_node):
                earliest = [ef for ef in self._earliest_topic_wikiFiles(topic_node) if ef != fname]
                if earliest:
                    tctx[topic_node.removeprefix("topic:")] = earliest
                    for ef in earliest:
                        union.setdefault(ef, "topic")
            topic_context[fname] = tctx

        # Drop thread/topic neighbors that are themselves seeds (avoid redundancy).
        for fname in seeds:
            #thread_neighbors[fname] = [n for n in thread_neighbors.get(fname, []) if union.get(n) != "seed"]
            topic_context[fname] = {
                t: [ef for ef in files if union.get(ef) != "seed"]
                for t, files in topic_context.get(fname, {}).items()
            }
        #return self._assemble_context(seeds, """thread_neighbors,""" topic_context)
        return self._assemble_context(seeds, topic_context)


def make_retriever(kind: str, WikiFiles: str, db: Chroma) -> BaseRetriever:
    if kind == "bm25":
        return Bm25Retriever(WikiFiles=WikiFiles)
    if kind == "vector":
        return VectorRetriever(db=db)
    return HybridRetriever(WikiFiles=WikiFiles, db=db)


# ─── RAGAS evaluation harness ────────────────────────────────────────
def make_judge():
    """Build the judge LLM. Called from main() AFTER require_api_key(), so a missing
    key gives a friendly message instead of a KeyError at import time."""
    return llm_factory(
        JUDGE_MODEL,
        client=OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL),
    )


judge = None  # built in main() after the API-key check

# Here is a reasonable starting metric: Grade on the main point(s), not exact coverage of every 
# detail. The grading notes are multipoint summaries, so a judge told to require *all*
# key points fails almost every real answer (0 pass) and gives you no signal to learn from.
# This version passes an answer that is consistent with the notes and captures their main
# point(s). Tuning it (e.g., require a "partial" level, demand every point, or try a stronger
# JUDGE_MODEL) is the Step 2.
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt=(
        "You are grading a retrieval-augmented answer against reference grading notes.\n"
        "Return 'pass' if the response is factually consistent with the grading notes and "
        "captures their main point(s) — even if it omits some minor details or is worded "
        "differently. Return 'fail' only if the response contradicts the notes, is "
        "unsupported by them, or misses the central point.\n"
        "Response: {response}\nGrading Notes: {grading_notes}"
    ),
    allowed_values=["pass", "fail"],
)


def load_dataset(inputs_path: Path) -> Dataset:
    dataset = Dataset(name="wiki_db_eval", backend="local/csv", root_dir="ragas_experiments")
    with open(inputs_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for sample in samples:
        dataset.append({"question": sample["question"], "grading_notes": sample["grading_notes"]})
    dataset.save()
    return dataset


def build_experiment(retriever: BaseRetriever, kind: str):
    @experiment()
    async def run_experiment(row):
        response = retriever.query(row["question"])
        score = correctness_metric.score(
            llm=judge,
            response=response,
            grading_notes=row["grading_notes"],
        )
        return {**row, "retriever": kind, "response": response, "score": score.value}

    return run_experiment


async def main():
    require_api_key()
    global judge
    judge = make_judge()
    parser = argparse.ArgumentParser(description="Evaluate a retriever with RAGAS.")
    parser.add_argument("retriever", choices=["bm25", "vector", "hybrid", "graph"])
    parser.add_argument("WikiFiles", nargs="?", default="WikiFiles")
    parser.add_argument("--inputs", default=str(Path(__file__).parent / "testInputs.json"))
    args = parser.parse_args()


    #annotated_path = sys.argv[1] if len(sys.argv) > 1 else "annotatedWikiFiles.json"
    annotated_path = "annotatedWikiFiles.json"
    print(f"Loading {annotated_path} ...")
    wikiFiles, body_by_filename = load_annotated(annotated_path)
    print(f"  {len(wikiFiles)} wiki files loaded.")
    db = build_or_load_db(args.WikiFiles, CHROMA_DIR)

    if args.retriever == "graph":
        hybrid = HybridRetriever(args.WikiFiles, db)
        print("Building knowledge graph ...")
        graph = build_graph(wikiFiles)
        print(f"  graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        retriever = GraphRetriever(graph=graph, hybrid=hybrid, body_by_filename=body_by_filename)
    else:
        retriever = make_retriever(args.retriever, args.WikiFiles, db)

    dataset = load_dataset(Path(args.inputs))
    print(f"Loaded {len(dataset)} questions. Evaluating with '{args.retriever}' retriever...")

    results = await build_experiment(retriever, args.retriever).arun(dataset)

    print(f"receiving result...")

    passes = sum(1 for r in results if r["score"] == "pass")
    print(f"Experiment complete: {passes}/{len(results)} passed.")

    results.save()
    csv_path = Path("ragas_experiments") / "experiments" / f"{results.name}.csv"
    print(f"Results saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
