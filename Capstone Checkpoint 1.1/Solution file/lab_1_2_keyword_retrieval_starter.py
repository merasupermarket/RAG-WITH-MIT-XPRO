"""Module 1: keyword retrieval over the wikipedia text with BM25.

Developer: Gaurav Singh, 08/24/2026
git- https://github.com/merasupermarket/RAG-WITH-MIT-XPRO

Setup
-----
1. create directory 'WikiFiles' in this folder and place the Wikipedia text files (one .txt per article) in it. 
2. for this lab, you can use the provided 2022_US_elections.txt file. 
3. You can also extract more Wikipedia articles using the provided WikiExtract.py script.



Run:  python lab_1_2_keyword_retrieval_starter.py <wikipedia_files_folder>

How it works 
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
3. Wikipedia data: Place the Wikipedia text files (one .txt per article) in a folder
   named 'WikiFiles' in this directory, or pass a folder path as the first
   argument. The folder MUST contain the .txt files.

Sample questions to try (over the Wikipedia corpus)
-----------------------------------------------
    "who was incumbent in 2022 election?"
        -> Grounded answer: The incumbent president in the 2022 elections was Joe Biden (D). (Source: 2022_US_elections.txt)
    "What were the main issues for the 2022 election?"
        ->  1. **Economy**: High consumer prices, inflation, and gas prices were significant concerns for voters, with Republicans blaming Biden's policies and Democrats linking them to global factors, including the COVID-19 pandemic and the Russian invasion of Ukraine.

            2. **Abortion**: Following the Supreme Court's ruling in Dobbs v. Jackson Women's Health Organization, abortion became a major issue, influencing several elections and driving Democratic voter turnout.

            3. **Crime and Gun Violence**: Mass shootings and rising crime rates contributed to voter concerns, with Republicans emphasizing crime rates and Democrats advocating for gun safety laws.

            4. **Democracy**: The integrity of democratic institutions and threats posed by election deniers and authoritarianism were highlighted by Democrats as key themes in their campaigns.

            5. **Climate Change**: Climate change emerged as a significant issue, with a majority of voters considering it a serious problem and some candidates actively campaigning on climate-related policies (source: 2022_US_elections.txt).
    "When did Howdy Modi happend in Taxas?"
        -> The provided articles do not contain information about "Howdy Modi" in Texas. Therefore, I cannot answer your question.
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

SYSTEM_PROMPT = """You are a helpful assistant for Political Survey Inc. \
You answer questions by drawing information exclusively from the wikipedia articles \
provided to you as context in each message.

Rules:
- If the answer can be found in the provided wikipedia articles, answer clearly and concisely.
- If the provided articles do not contain enough information to answer the question, \
say so explicitly and do not speculate or use outside knowledge.
- Cite the source filename in your answer using the filename shown in the context.
- Do not answer questions that are unrelated to the content of the provided wikipedia articles."""

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
    print("Chat over the wikipedia articles. Type your question; 'exit'/'quit' to stop.\n")
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
        return f"Context (wikipedia articles):\n{context}\n\nQuestion: {query}"

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
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _STOPWORDS]


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

if __name__ == "__main__":
    require_api_key()
    wiki_dir = sys.argv[1] if len(sys.argv) > 1 else "WikiFiles"
    Bm25Retriever(wiki_dir=wiki_dir).chat()