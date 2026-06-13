from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from data_doctor.agents.Main import MainAgent

load_dotenv()

app = FastAPI(title="Data Doctor API")

main_agent = MainAgent()
graph = main_agent.build_graph()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    response = graph.invoke({"messages": messages})
    reply = response["messages"][-1].content
    return ChatResponse(reply=reply)
