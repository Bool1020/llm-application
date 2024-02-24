from apps.handle.chat.api_rpc import chat, model_message
from apps.handle.knowledge.knowledge_utils import load_knowledge
from apps.handle.retrieval.retrieval_utils import Search
from apps.handle.mylang.chain_utils import llm
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.render import format_tool_to_openai_function
from langgraph.prebuilt import ToolExecutor
from langchain.chat_models import ChatBaichuan
from apps.handle.mylang.chain_utils import CustomChat
import os
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
)


if __name__ == '__main__':
    db = load_knowledge('test')
    search = Search(db)

    from apps.handle.mylang.agent_utils import AgenticRAG

    rag = AgenticRAG(search)

    inputs = {
        "messages": [
            HumanMessage(
                content="What does Lilian Weng say about the types of agent memory?"
            )
        ]
    }
    for output in rag(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
