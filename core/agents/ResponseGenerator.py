from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from core.agents import State
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage


class Response(BaseModel):
    response : str

def generate_response(state: State):
    """
    Generate a response based on the information processed by previous agents.
    """
    message = state.get("messages", [])[-1]
    previous_agent = state.get("agent", "Unknown Agent")
    sys_prompt = (
        "You are a response agent. Based on the information provided by the previous agent, "
        "generate a response to the user's request.\n"
        "# Previous Agent Information\n"
        "{previous_agent}\n"
        "# User's Request\n"
        "{message}\n"
        "# Previous Conversation\n"
        "{conversation}\n"
        "# Response Format\n"
        "\n{{\n"
        "    \"response\": \"<your response here>\"\n"
        "}}\n"
    )
#"EHR", "Dianosis", "ML"
    information = ""
    if previous_agent == "EHRAgent":
        information = "EHRAgent answered with the following results:\n"
        information += str(state.get("data_response",""))
    elif previous_agent == "DiagnosisAgent":
        information = "DiagnosisAgents found the following documents:\n"
        information += "\n".join(state.get("documents", []))
    elif previous_agent == "MLAgent":
        information = f"MLAgent gave the following prediction for {state.get('target')}:\n"
        information += str(state.get("prediction"))
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        timeout=10
    )
    prompt = PromptTemplate(
        template=sys_prompt,
        input_variables=["previous_agent", "message"],
    )
    chain = prompt | llm.with_structured_output(
        Response
    )
    response: Response = chain.invoke({
        "previous_agent": information,
        "message": message,
        "conversation": "\n".join([f"{msg.type}: {msg.content}" for msg in state.get("messages", [])[:-1]])
    })
    return {
        "response": response.response,
        "messages": state.get("messages", []) + [AIMessage(content=response.response)]
    }

class ResponseGenerator:
    def __init__(self):
        self.name = "ResponseGenerator"
        self.description = (
            "This agent processes the information from previous agents and provides a response to the user."
        )

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("generate_response", generate_response)
        graph.add_edge(START, "generate_response")
        graph.add_edge("generate_response", END)
        return graph.compile()
