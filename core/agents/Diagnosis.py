from langgraph.graph import StateGraph, START, END
from core.agents import State
from core.database import VectorStore

def get_documents(filepaths):
    # Get documents files and format it into a string
    documents = {}
    for filename in filepaths:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            documents[filename] = content
    documents_str = "\n\n".join([f"Document: {filename}\nContent:\n{content}" for filename, content in documents.items()])
    return documents_str


def search_documents(state: State):
    message = state.get("messages", [])[-1]
    vector_store = VectorStore("diagnosis")
    points = vector_store.search(message.content, limit=5)
    documents = [p.payload['source'] for p in points.points]
    documents_str = get_documents(documents)
    return {
        "documents": documents_str
    }

class Diagnosis:
    """
    Agent that extracts information from previous clinical brief diagnosis.
    """
    def __init__(self):
        self.name = "DiagnosisAgent"
        self.description = (
            "This agent extracts information from previous clinical brief diagnosis."
            "It is used when the user requests information not present on the health records"
        )

    def build_graph(self):
        graph = StateGraph(State)
        graph.add_node("search_documents", search_documents)
        graph.add_edge(START, "search_documents")
        graph.add_edge("search_documents", END)
        return graph.compile()
