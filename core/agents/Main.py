from typing import Literal
from core.agents import Planner, EHR, Diagnosis, ML, ResponseGenerator
from core.agents import State
from langgraph.graph import StateGraph, START, END


def get_agents_descriptions(agents):
    """
    Get the descriptions of the agents in a formatted string.
    """
    descriptions = []
    for agent in agents:
        descriptions.append(f"**{agent.name}**")
        descriptions.append(agent.description)
        descriptions.append("\n")
    return "\n".join(descriptions)

def next_agent(state) -> Literal["EHRAgent", "DiagnosisAgent", "MLAgent", "ResponseGenerator", END]:
    return state.get("agent", END)

class MainAgent:
    """
    MainAgent coordinates the execution of other agents.
    """
    def __init__(self):
        self.name = "MainAgent"
        self.description = "This agent coordinates the execution of other agents."
        self.agents = self.get_agents()

    def get_agents(self):
        """
        Get the list of agents that this MainAgent will coordinate.
        """
        return [EHR(), Diagnosis(), ML()]

    def build_graph(self):
        """
        Build the execution graph for the MainAgent.
        """
        graph = StateGraph(State)
        graph.add_node("PlannerAgent", Planner().build_graph(get_agents_descriptions(self.agents)))
        graph.add_node("ResponseGenerator", ResponseGenerator().build_graph())
        graph.add_edge(START, "PlannerAgent")
        graph.add_conditional_edges("PlannerAgent", next_agent)
        for agent in self.agents:
            graph.add_node(agent.name, agent.build_graph())
            graph.add_edge(agent.name, "ResponseGenerator")
        graph.add_edge("ResponseGenerator", END)
        return graph.compile()
