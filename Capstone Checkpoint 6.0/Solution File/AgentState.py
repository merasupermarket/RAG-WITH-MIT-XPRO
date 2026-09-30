# ─── Provided: tool-using agent (LangGraph: plan -> retrieve/by_date/clarify/answer) ──
from langchain_protocol import TypedDict


class _AgentState(TypedDict):
    conversation_history: list[dict]
    clarification_history: list[str]
    pending_queries: list[str]
    pending_file_name: str
    executed_queries: list[str]
    wikifile_bodies: dict[str, str]
    iterations: int
    mode: str
    next_action: str
    clarification_question: str
    answer: str
    done: bool