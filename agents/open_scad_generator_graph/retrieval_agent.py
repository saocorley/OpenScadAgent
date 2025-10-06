import logging

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI

from agents.agent_utils.parsing_utils import parse_retrieval_output

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Load system prompt once at module level
with open("agents/prompts/retrieval_agent.txt", "r") as file:
    RETRIEVAL_SYSTEM_PROMPT = file.read()

# Create LLM object once at module level
RETRIEVAL_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retrieval_agent(state) -> dict:
    """This agent is responsible for retrieving information from the official bpy documentation implementation."""
    """We will always assume that the first assignment is the relevant one. We will pop it when we are done."""
    sys_message = SystemMessage(content=RETRIEVAL_SYSTEM_PROMPT)
    retriever_tool = create_retriever_tool(
        state["bm25"],
        "retrieval_tool",
        "Retrieves information from the official OpenSCAD documentation",
        document_prompt=None,
        document_separator="\n\n",
    )
    tools = [retriever_tool]
    retrieval_query = state["assignments"][0][1]

    messages = [sys_message] + [HumanMessage(content=retrieval_query)]
    llm_with_tools = RETRIEVAL_LLM.bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    logger.info("Tool calls detected: %s", len(response.tool_calls or []))

    # Execute the tool calls and get results
    if response.tool_calls:
        tool_messages = []

        for tool_call in response.tool_calls:
            logger.info(
                f"Executing tool: {tool_call['name']} with args: {tool_call['args']}"
            )
            # Find and invoke the tool
            try:
                tool_result = retriever_tool.invoke(tool_call["args"])
                tool_message = ToolMessage(
                    content=str(tool_result), tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
                break
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                tool_message = ToolMessage(
                    content=f"Tool execution failed: {str(e)}",
                    tool_call_id=tool_call["id"],
                )
                tool_messages.append(tool_message)

        # Add tool messages to state
        parsed_output = parse_retrieval_output(retrieval_query, tool_messages)
        # context that will be fed to the coding agent
        state["lastest_context"] = parsed_output

        # append the parsed output to the messages, for debugging purposes
        state["messages"].append(
            AIMessage(
                content="Retrieved information for query: "
                + retrieval_query
                + "\n\n"
                + parsed_output[:100]
                + "..."
            )
        )

    else:
        logger.info("No tool calls to execute")
    # Move completed assignment to completed_assignments list
    completed_assignment = state["assignments"].pop(0)
    state["completed_assignments"].append(completed_assignment)
    return state
