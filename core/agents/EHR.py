from langgraph.graph import StateGraph, START, END
from core.utils import get_columns
from core.agents import State
import pandasai as pai
from pandasai_litellm import LiteLLM
import os


def interact_data(state: State) -> State:
    pai.config.set({
        "llm" : LiteLLM(model="gpt-4o-mini")
    })
    message = state.get("messages", [])[-1]
    df = pai.read_csv(os.getenv("EHR_DATA_PATH"))
    response = df.chat(message.content)
    return {
        "data_response": response
    }

class EHR:
    def __init__(self):
        self.name = "EHRAgent"
        self.description = (
            "This agent processes Electronic Health Records (EHR) data and provides predictions."
            "It is used when the user requests information from EHR data."
            "EHR data contains the following fields: "
            f"{get_columns()}"
        )

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("interact_data", interact_data)
        graph.add_edge(START, "interact_data")
        graph.add_edge("interact_data", END)
        return graph.compile()
