from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent: str
    prediction : float | str
    data_response: str | None
    documents: list[str] | None
    target: str | None
    data: dict | None
