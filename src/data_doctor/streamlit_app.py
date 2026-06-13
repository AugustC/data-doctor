import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Data Doctor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask Data Doctor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                json={"messages": st.session_state.messages},
                timeout=60,
            )
            response.raise_for_status()
            reply = response.json()["reply"]

        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
