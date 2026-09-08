"""Lab 2.2 (SOLUTION) — hybrid retrieval: fuse BM25 keyword + vector search.

The answers are shown in the console from BM25, Vector and hybrid retrieval as the same time
The hybrid retriever combines the two rankings so you get the best of each.

Developer: Gaurav Singh, 08/31/2026
git- https://github.com/merasupermarket/RAG-WITH-MIT-XPRO

Setup
-----
1. Create directory 'WikiFiles' and place the Wikipedia text files (one .txt per article) in it. 
2. for this lab, you can use the provided Apollo_11.txt file. 
3. You can also extract more Wikipedia articles using the provided WikiExtract.py script.
4. Ensure that you have completed the program's one-time environment and
   OpenRouter API key setup, and activate the configured environment.
5. Install any additional dependencies required for this lab, if they are
   not already available:
       pip install python-dotenv langchain-openai langchain-core langchain-chroma rank-bm25  
6. The vector DB is persisted to ./chroma_db (delete it to rebuild).

Run:  python lab_2_2_hybrid_retrieval_starter.py 

How it works 
-----------
Hybrid retrieval runs both and combines their rankings so you get the
best of each. However, BM25 scores are "higher = better", and Chroma returns a
DISTANCE where "lower = better". So the trick to combining them is to normalize each to [0,1] and INVERT the
vector side before taking a weighted sum.


Sample questions to try (example expected behavior)
-----------------------------------------------
    "who was the head of the state when men landed on moon?"
        -> Grounded answer: The head of state at the time the Apollo 11 crew landed on the Moon (20 July 1969) was **President Richard Nixon**
    "When did Apollo One failed?"
        -> Provides the date of the failure.
    " How many days passed between the selection of lunar orbit rendezvous as the Apollo mission mode and the fatal fire during the launchpad test?"
        -> The Apollo program officially adopted **lunar‑orbit rendezvous (LOR)** as its mission mode in **July 1962** when NASA’s James Webb announced that the new approach would be used【Apollo 11.txt†L384-L389】.  
        A little more than four and a half years later the program was halted by the **Apollo 1 launch‑pad fire on 27 January 1967**, which killed the crew of Grissom, White and Chaffee【Apollo 11.txt†L436-L442】.  
            Counting the days from 1 July 1962 (the first day of the month in which LOR was announced) to 27 January 1967 gives:

            * 1 Jul 1962 → 1 Jul 1963 = 365 days  
            * 1 Jul 1963 → 1 Jul 1964 = 366 days (leap year)  
            * 1 Jul 1964 → 1 Jul 1965 = 365 days  
            * 1 Jul 1965 → 1 Jul 1966 = 365 days  
            * 1 Jul 1966 → 27 Jan 1967 = 210 days  

            **Total = 365 + 366 + 365 + 365 + 210 = 1,671 days**

            So **1,671 days** elapsed between the selection of LOR as the Apollo mission mode and the fatal Apollo 1 fire
            
    *****************************************************
    ****Compare the retrievers on the same questions*****
    *****************************************************

        -> what was the height of Saturn V?

        Answer from BM25
        : The provided Apollo 11 WikiFile does not include the height of the Saturn V launch vehicle.

        Answer from Vector
        : The Saturn V rocket used for Apollo 11 was **363 feet (about 111 meters) tall**.

        Answer from Hybrid
        : The provided Apollo 11 WikiFile does not include the height of the Saturn V launch vehicle.

"""
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
TOP_K = 5
NUM_RETRIEVED = 4          # WikiFiles sent to the LLM as context.
CANDIDATE_POOL = 10        # Candidates pulled from EACH retriever before fusion.
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

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
    print("Chat over the WikiFiles. Type your question; 'exit'/'quit' to stop.\n")
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


# ─── Keyword side (from Lab 1.2) ─────────────────────────────────────
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


# ─── Vector side  ──────────────────────────────────────
def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_or_load_db(WikiFiles: str, chroma_dir: str = CHROMA_DIR) -> Chroma:
    if os.path.isdir(chroma_dir) and os.listdir(chroma_dir):
        print(f"Loading existing vector DB from {chroma_dir}/")
        return Chroma(persist_directory=chroma_dir, embedding_function=get_embeddings_model())
    print("Building vector DB (first run — embedding the WikiFiles)...")
    docs = []
    for fname in sorted(os.listdir(WikiFiles)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(WikiFiles, fname), encoding="utf-8", errors="replace") as fh:
            docs.append(Document(page_content=fh.read(), metadata={"source": fname}))
    db = Chroma.from_documents(docs, get_embeddings_model(), persist_directory=chroma_dir)
    print(f"  Indexed {len(docs)} WikiFiles into {chroma_dir}/")
    return db


def _normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores 
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH 
 normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)  # all equal → neutral
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm


class Bm25Retriever(BaseRetriever):
    def __init__(self, wiki_dir: str, top_k: int = TOP_K, **kwargs):
        super().__init__(**kwargs)
        paths = sorted(
            os.path.join(wiki_dir, f) for f in os.listdir(wiki_dir) if f.endswith(".txt")
        )
        self._paths = [os.path.basename(p) for p in paths]
        self._contents = []
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as fh:
                self._contents.append(fh.read())
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        self._top_k = top_k
        print(f"Indexed {len(self._paths)} wikipedia articles.")

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:        
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
        results = self.getTopK(query, self._top_k)
        return "\n\n---\n\n".join(
            f"[{filename}]\n{content}" for filename, content, _ in results
        )

class HybridRetriever(BaseRetriever):
    def __init__(self, WikiFiles: str, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._keyword_retriever = Bm25Retriever(wiki_dir=WikiFiles, top_k=TOP_K, **kwargs)
        # Keyword index over the same WikiFiles (kept in memory).
        paths = sorted(
            os.path.join(WikiFiles, f) for f in os.listdir(WikiFiles) if f.endswith(".txt")
        )
        self._paths = [os.path.basename(p) for p in paths]
        self._contents = []
        for p in paths:
            with open(p, encoding="utf-8", errors="replace") as fh:
                self._contents.append(fh.read())
        self._bm25 = BM25Okapi([tokenize(doc) for doc in self._contents])
        self._db = db
        print(f"Hybrid retriever ready over {len(self._paths)} WikiFiles.")

    def chat(self) -> None:
        print("Chat over the WikiFiles. Type your question; 'exit'/'quit' to stop.\n")
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue

            keyword_context = self._keyword_retriever.retrievedContext(user_input)
            hybrid_context = self.retrievedContext(user_input)
            vector_context = self._vector_context(user_input)

            #print("\nBM25 results:\n")
            #print(keyword_context)
            #print("\nVector results:\n")
            #print(vector_context)
            #print("\nHybrid results:\n")
            #print(hybrid_context)

            bm25_answer = self._keyword_retriever.query(user_input)
            vector_answer = self._answer_from_context(user_input, vector_context)
            hybrid_answer = self.query(user_input)

            print(f"\nAnswer from BM25\n: {bm25_answer}\n")
            print(f"\nAnswer from Vector\n: {vector_answer}\n")
            print(f"\nAnswer from Hybrid\n: {hybrid_answer}\n")

    def _answer_from_context(self, question: str, context: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_user_message(question, context)),
        ]
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def _vector_context(self, query: str) -> str:
        docs = self._db.similarity_search(query, k=NUM_RETRIEVED)
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )

    def _bm25_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._paths[i], self._contents[i], scores[i]) for i in top]

    def _vector_topk(self, query: str, k: int) -> list[tuple[str, str, float]]:
        results = self._db.similarity_search_with_score(query, k=k)
        return [(d.metadata.get("source", "unknown"), d.page_content, s) for d, s in results]

    def getTopK(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Fuse the two retrievers: Pull a candidate pool from each, normalize, and
        combine with a weighted sum. BM25 is higher-is-better and vector distance is
        lower-is-better, so the vector side is normalized with invert=True. A doc
        missing from one retriever simply contributes 0 from that side."""
        bm = self._bm25_topk(query, CANDIDATE_POOL)
        vec = self._vector_topk(query, CANDIDATE_POOL)

        content_by_name: dict[str, str] = {}
        bm_norm: dict[str, float] = {}
        vec_norm: dict[str, float] = {}

        if bm:
            for (name, content, _), val in zip(bm, _normalize([s for _, _, s in bm])):
                content_by_name[name] = content
                bm_norm[name] = val
        if vec:
            for (name, content, _), val in zip(vec, _normalize([d for _, _, d in vec], invert=True)):
                content_by_name[name] = content
                vec_norm[name] = val

        fused = [
            (name, content, WEIGHT_BM25 * bm_norm.get(name, 0.0) + WEIGHT_VECTOR * vec_norm.get(name, 0.0))
            for name, content in content_by_name.items()
        ]
        fused.sort(key=lambda t: t[2], reverse=True)
        return fused[:k]

    def retrievedContext(self, query: str) -> str:
        results = self.getTopK(query, NUM_RETRIEVED)
        return "\n\n---\n\n".join(f"[{name}]\n{content}" for name, content, _ in results)


if __name__ == "__main__":
    require_api_key()
    WikiFiles = sys.argv[1] if len(sys.argv) > 1 else "WikiFiles"
    db = build_or_load_db(WikiFiles, CHROMA_DIR)
    HybridRetriever(WikiFiles=WikiFiles, db=db).chat()