from core.agents import State
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class Plan(BaseModel):
    next_agent: str

def get_generate_plan(agents_descriptions):
    def generate_plan(state: State) -> dict:
        message = state.get("messages", [])[-1]
        sys_prompt = (
            "You are a router agent. Select which of the following agents is most suitable to handle the user's request"
            "# Agents\n"
            "{agents_descriptions}\n"
            "# User's request\n"
            "{message}\n"
            "# Previous Conversation\n"
            "{conversation}\n"
            "# Rules\n"
            "- If the user is sending a small talk message, return 'ResponseGenerator' agent.\n"
            "- If the user is asking for health-related information, return 'Diagnosis' agent.\n"
            "# JSON Output Format"
            "\n{{\n"
            "    \"next_agent\": \"<name of the agent>\"\n"
            "}}\n"
        )
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
            timeout=10
        )
        prompt = PromptTemplate(
            template=sys_prompt,
            input_variables=["agents_descriptions", "message", "conversation"],
        )
        chain = prompt | llm.with_structured_output(
            Plan
        )
        response : Plan = chain.invoke({
            "agents_descriptions": agents_descriptions,
            "message": message,
            "conversation": "\n".join([f"{msg.type}: {msg.content}" for msg in state.get("messages", [])[:-1]])
        })
        return {
            "agent": response.next_agent
        }
    return generate_plan

class Planner:
    """
    Planner agent that generates plans based on the input message
    """
    def __init__(self):
        self.name = "PlannerAgent"
        self.description = "This agent generates plans based on the input message."

    def build_graph(self, agents_descriptions):
        graph = StateGraph(State)
        graph.add_node("generate_plan", get_generate_plan(agents_descriptions))
        graph.add_edge(START, "generate_plan")
        graph.add_edge("generate_plan", END)
        return graph.compile()
