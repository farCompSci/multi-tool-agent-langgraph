import streamlit as st
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

load_dotenv()

from main_graph import build_supervisor_graph
from helpers.supervisor.long_term_memory import LongTermMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs.txt', encoding='utf-8'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

ltm = LongTermMemory()

st.set_page_config(page_title="AI Agent with Long-Term Memory", layout="wide")
st.title("AI Agent with Long-Term Memory")

if "supervisor_graph" not in st.session_state:
    st.session_state.supervisor_graph = build_supervisor_graph()
    st.session_state.graph_executor = st.session_state.supervisor_graph.with_config(
        {"configurable": {"thread_id": "streamlit_session"}}
    )
    st.session_state.messages = []
    st.session_state.last_message_count = 0
    logger.info("Initialized new Streamlit session with supervisor graph")


def log_message(message_type, content, user_id="streamlit_user"):
    """Log messages to file with timestamp and user info"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{user_id}] {message_type}: {content}")


def display_messages():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, SystemMessage):
            st.info(f"System: {msg.content}")
        elif isinstance(msg, ToolMessage):
            st.code(f"Tool Output: {msg.content}", language="text")


user_input = st.chat_input("Type your message here...")

if user_input:
    log_message("USER_INPUT", user_input)

    user_message = HumanMessage(content=user_input)
    st.session_state.messages.append(user_message)
    st.chat_message("user").write(user_input)

    with st.spinner("Thinking..."):
        try:
            logger.info(f"Processing user input: {user_input[:100]}...")

            graph_output = st.session_state.graph_executor.invoke(
                {"messages": [user_message]}
            )

            all_graph_messages = graph_output.get("messages", [])
            new_messages = all_graph_messages[st.session_state.last_message_count:]

            filtered_new_messages = []
            for msg in new_messages:
                if isinstance(msg, HumanMessage) and msg.content == user_input:
                    continue
                filtered_new_messages.append(msg)

            display_queue = []
            for i, msg in enumerate(filtered_new_messages):
                if isinstance(msg, ToolMessage):
                    # Look ahead to see if the next message is an AI explanation
                    if i + 1 < len(filtered_new_messages) and isinstance(filtered_new_messages[i + 1], AIMessage):
                        continue  # Skip this tool message if an AI explanation follows
                display_queue.append(msg)

            for msg in display_queue:
                if isinstance(msg, AIMessage):
                    st.chat_message("assistant").write(msg.content)
                    log_message("AI_RESPONSE", msg.content)
                elif isinstance(msg, SystemMessage):
                    st.info(f"System: {msg.content}")
                    log_message("SYSTEM_MESSAGE", msg.content)
                elif isinstance(msg, ToolMessage):
                    st.code(f"Tool Output: {msg.content}", language="text")
                    log_message("TOOL_OUTPUT", msg.content)

                st.session_state.messages.append(msg)

            st.session_state.last_message_count = len(all_graph_messages)

            logger.info(f"Successfully processed user input. Total messages: {st.session_state.last_message_count}")

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            log_message("ERROR", str(e))
            st.session_state.messages.append(AIMessage(content=f"Sorry, I encountered an error: {e}"))

display_messages()

st.sidebar.header("Controls")

if st.sidebar.button("Clear Chat History"):
    logger.info("Clearing chat history")
    st.session_state.messages = []
    st.session_state.last_message_count = 0
    st.session_state.graph_executor = st.session_state.supervisor_graph.with_config(
        {"configurable": {"thread_id": "streamlit_session"}}
    )
    st.rerun()

if st.sidebar.button("Clear All Long-Term Memories (DANGER!)"):
    try:
        logger.warning("Clearing all long-term memories")
        ltm.vectorstore.delete(ids=ltm.vectorstore.get()['ids'])
        st.sidebar.success("All long-term memories cleared!")
        st.session_state.messages = []
        st.session_state.last_message_count = 0
        st.session_state.graph_executor = st.session_state.supervisor_graph.with_config(
            {"configurable": {"thread_id": "streamlit_session"}}
        )
        logger.info("Successfully cleared all memories and reset session")
        st.rerun()
    except Exception as e:
        error_msg = f"Error clearing memories: {e}"
        st.sidebar.error(error_msg)
        log_message("ERROR", error_msg)

st.sidebar.header("Logging Info")
st.sidebar.info("All interactions are logged to `logs.txt`")