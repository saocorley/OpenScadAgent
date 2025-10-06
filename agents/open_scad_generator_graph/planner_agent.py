import logging
from typing import List, Literal

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Load system prompt once at module level
with open("agents/prompts/planner_agent.txt", "r") as file:
    PLANNER_SYSTEM_PROMPT = file.read()


class Assignment(BaseModel):
    agent: Literal["retrieval agent", "coding agent", "assembly agent"]
    task: str


class PlanWithAssignments(BaseModel):
    reasoning: str
    assignments: List[Assignment]
    is_complete: bool  # True if the planner believes the task is finished


# Create LLM objects once at module level (after class definitions)
PLANNER_LLM = ChatOpenAI(model="gpt-4o", temperature=0)
PLANNER_STRUCTURED_LLM = PLANNER_LLM.with_structured_output(
    PlanWithAssignments, method="function_calling"
)


def planner_agent(state) -> dict:
    """
    The planner agent is responsible for:
    1. Understanding the user's question
    2. Think of ways of breaking down the shape of the object into simpler shapes.
    3. Formulate an idea of how to create this shape in OpenSCAD.
    """

    sys_message = SystemMessage(content=PLANNER_SYSTEM_PROMPT)

    # Always start with system prompt and original query
    messages = state.get("messages") or [
        sys_message,
        HumanMessage(content=state["original_query"]),
    ]

    # Add information about completed work if any
    completed_assignments = state.get("completed_assignments", [])
    if completed_assignments:
        completed_work = "\\n".join(
            [f"✓ {agent}: {task}" for agent, task in completed_assignments]
        )
        work_summary = AIMessage(content=f"COMPLETED WORK SO FAR:\\n{completed_work}")
        messages.append(work_summary)

    response = PLANNER_STRUCTURED_LLM.invoke(messages)

    # Extract both reasoning and structured assignments from single response
    plan_text = response.reasoning
    assignments = [
        (assignment.agent, assignment.task) for assignment in response.assignments
    ]

    # Convert structured response to AIMessage for message history
    plan_message = AIMessage(
        content=f"Plan: {plan_text}\nAssignments: {assignments}\nComplete: {response.is_complete}"
    )

    messages.append(plan_message)
    state["messages"] = messages
    state["plan"] = plan_text
    state["assignments"] = assignments
    state["is_complete"] = response.is_complete
    return state
