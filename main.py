from core.agents.Main import MainAgent
from dotenv import load_dotenv
from termcolor import colored

if __name__ == "__main__":
    load_dotenv()
    main_agent = MainAgent()
    graph = main_agent.build_graph()
    print("Agent Initialized Successfully!")
    USER = colored("User", "green", attrs=["bold"])
    ASSISTANT = colored("Assistant", "blue", attrs=["bold"])
    messages = []

    while True:
        user_input = input(f"{USER}: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print(f"{ASSISTANT}: Goodbye!")
            break
        messages.append({"role": "user", "content": user_input})
        response = graph.invoke({"messages": messages})
        messages.append({"role": "assistant", "content": response['messages'][-1].content})
        print(f"{ASSISTANT}: {response['messages'][-1].content}")
