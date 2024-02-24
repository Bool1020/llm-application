import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
import json
from .chain_utils import CustomChat
from langchain.chat_models import ChatOpenAI


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


class AgenticRAG(object):
    def __init__(self, search):
        tool = create_retriever_tool(
            search.retriever,
            'retriever',
            'Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.'
        )
        self.tools = [tool]
        self.tool_executor = ToolExecutor(self.tools)
        self.graph = StateGraph(AgentState)

        # Edges

        def should_retrieve(state):
            print("---DECIDE TO RETRIEVE---")
            messages = state["messages"]
            last_message = messages[-1]
            if "function_call" not in last_message.additional_kwargs:
                print("---DECISION: DO NOT RETRIEVE / DONE---")
                return "end"
            else:
                print("---DECISION: RETRIEVE---")
                return "continue"

        def grade_documents(state):
            print("---CHECK RELEVANCE---")

            class grade(BaseModel):
                binary_score: str = Field(description="Relevance score 'yes' or 'no'")

            # Tool
            grade_tool_oai = convert_to_openai_tool(grade)

            # llm = CustomChat()
            llm = ChatOpenAI(temperature=0.0)
            llm_with_tool = llm.bind(
                tools=[convert_to_openai_tool(grade_tool_oai)],
                tool_choice={"type": "function", "function": {"name": "grade"}},
            )

            # Parser
            parser_tool = PydanticToolsParser(tools=[grade])

            # Prompt
            prompt = PromptTemplate(
                template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the retrieved document: \n\n {context} \n\n
                Here is the user question: {question} \n
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
                input_variables=["context", "question"],
            )

            # Chain
            chain = prompt | llm_with_tool | parser_tool

            messages = state["messages"]
            last_message = messages[-1]

            question = messages[0].content
            docs = last_message.content

            score = chain.invoke(
                {"question": question,
                 "context": docs}
            )

            grade = score[0].binary_score

            if grade == "yes":
                print("---DECISION: DOCS RELEVANT---")
                return "yes"

            else:
                print("---DECISION: DOCS NOT RELEVANT---")
                print(score[0].binary_score)
                return "no"

        # Nodes

        def agent(state):
            print("---CALL AGENT---")
            messages = state["messages"]
            functions = [format_tool_to_openai_function(t) for t in self.tools]
            # llm = CustomChat()
            llm = ChatOpenAI(temperature=0.0)
            model = llm.bind_functions(functions)
            response = model.invoke(messages)
            return {"messages": [response]}

        def retrieve(state):
            print("---EXECUTE RETRIEVAL---")
            messages = state["messages"]
            last_message = messages[-1]
            action = ToolInvocation(
                tool=last_message.additional_kwargs["function_call"]["name"],
                tool_input=json.loads(
                    last_message.additional_kwargs["function_call"]["arguments"]
                ),
            )
            response = self.tool_executor.invoke(action)
            function_message = FunctionMessage(content=str(response), name=action.tool)
            return {"messages": [function_message]}

        def rewrite(state):
            print("---TRANSFORM QUERY---")
            messages = state["messages"]
            question = messages[0].content
            msg = HumanMessage(
                content=f""" \n 
            Look at the input and try to reason about the underlying semantic intent / meaning. \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Formulate an improved question: """,
            )
            # llm = CustomChat()
            llm = ChatOpenAI(temperature=0.0)
            response = llm.invoke([msg])
            return {"messages": [response]}

        def generate(state):
            print("---GENERATE---")
            messages = state["messages"]
            question = messages[0].content
            last_message = messages[-1]
            question = messages[0].content
            docs = last_message.content
            prompt = PromptTemplate(
                template="""
                    下面我会给你一段参考资料，请以此回答我的问题
                    {context}
                
                    问题：{question}
                    """,
                input_variables=["context", "question"],
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # llm = CustomChat()
            llm = ChatOpenAI(temperature=0.0)
            rag_chain = prompt | llm | StrOutputParser()
            response = rag_chain.invoke({"context": docs, "question": question})
            return {"messages": [response]}

        self.graph.add_node("agent", agent)  # agent
        self.graph.add_node("retrieve", retrieve)  # retrieval
        self.graph.add_node("rewrite", rewrite)  # retrieval
        self.graph.add_node("generate", generate)  # retrieval

        self.graph.set_entry_point("agent")

        self.graph.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_retrieve,
            {
                # Call tool node
                "continue": "retrieve",
                "end": END,
            },
        )

        self.graph.add_conditional_edges(
            "retrieve",
            grade_documents,
            {
                "yes": "generate",
                "no": "rewrite",
            },
        )
        self.graph.add_edge("generate", END)
        self.graph.add_edge("rewrite", "agent")

        # Compile
        self.app = self.graph.compile()

    def __call__(self, inputs):
        return self.app.stream(inputs)


def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate(
        [
            (
                'system',
                system_prompt
            ),
            MessagesPlaceholder(variable_name='messages'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': HumanMessage(content=result['output'], name=name)}


# class RAG(object):
#     def __init__(self, search):
#         self.search = search
#         from langchain.tools import tool
#
#         @tool('retriever')
#         def retriever(query: str) -> str:
#             docs = self.search.retriever(query)
#             return "\n\n".join(doc.page_content for doc in docs)
#
#
#         tool = create_retriever_tool(
#             search.retriever,
#             'retriever',
#             'Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.'
#         )
#         self.tools = [tool]
#         self.tool_executor = ToolExecutor(self.tools)
#         self.graph = StateGraph(AgentState)

class SelfRAG(AgenticRAG):
    pass

