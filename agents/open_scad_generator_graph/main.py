import logging
from typing import Dict, List

import dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, START, StateGraph

from agents.agent_utils.prepare_db import prepare_vector_database

# Environment will be loaded inside main() to keep imports at top-level
from agents.agent_utils.printing_utils import print_conversation_trace
from agents.open_scad_generator_graph.assembly_agent import assembly_agent
from agents.open_scad_generator_graph.coding_agent import coding_agent

# Import individual agents
from agents.open_scad_generator_graph.planner_agent import planner_agent
from agents.open_scad_generator_graph.retrieval_agent import retrieval_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class State(Dict):
    original_query: str
    messages: List[
        BaseMessage
    ]  # list of complete history. Useful for debugging, but not used internally.
    plan: str  # The plan of the agent. Not used internally.
    lastest_context: str  # Only keep one context at a time.
    vectors: InMemoryVectorStore
    bm25: BM25Retriever
    assignments: List[tuple]
    completed_assignments: List[tuple]
    is_complete: bool
    parts: List[dict]
    assembly_result: dict
    run_output_dir: str  # Persist output directory across nodes


def build_graph():
    """Builds the graph for the agent"""

    def route_after_planner(state: State) -> str:
        """Route to the appropriate agent based on planner assignments"""
        # Check if planner marked the work as complete
        if state.get("is_complete", False):
            logger.info("Planner marked work as complete, going to END")
            return "END"

        assignments = state.get("assignments", [])
        if not assignments:
            logger.info("No assignments from planner, going back to planner")
            return "planner"

        # Map planner agent names to graph node names
        agent_mapping = {
            "retrieval agent": "retrieval",
            "coding agent": "coding",
            "assembly agent": "assembly",
        }

        # Check if we have any assignments
        if assignments:
            agent, task = assignments[0]  # Take the first assignment
            # Map the agent name to the graph node name
            node_name = agent_mapping.get(agent.strip(), "END")
            logger.info(f"Routing to {node_name} for task: {task}")
            return node_name

        # If no assignments, go back to planner
        logger.info("No assignments found, going back to planner")
        return "planner"

    graph = StateGraph(State)
    graph.add_node("planner", planner_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("coding", coding_agent)
    graph.add_node("assembly", assembly_agent)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retrieval": "retrieval",
            "coding": "coding",
            "assembly": "assembly",
            "END": END,
            "planner": "planner",
        },
    )
    graph.add_conditional_edges(
        "retrieval",
        route_after_planner,
        {
            "retrieval": "retrieval",
            "coding": "coding",
            "assembly": "assembly",
            "END": END,
            "planner": "planner",
        },
    )
    graph.add_conditional_edges(
        "coding",
        route_after_planner,
        {
            "retrieval": "retrieval",
            "coding": "coding",
            "assembly": "assembly",
            "END": END,
            "planner": "planner",
        },
    )
    graph.add_edge("assembly", END)

    return graph.compile()


def main():
    """Main function"""
    # Load environment variables BEFORE importing agents execute
    dotenv.load_dotenv()
    logger.info("Building graph")
    graph = build_graph()
    logger.info("Preparing vector database")
    vector_store, bm25_retriever = prepare_vector_database(
        ["bibliography/wiki_primitives.txt", "bibliography/wiki_transformations.txt"]
    )
    # Create a per-run output directory under outputs/coder outputs/<short_id>
    import os
    import uuid

    base_outputs_dir = "/home/scorley/OpenScadAgent/outputs/coder_outputs"
    os.makedirs(base_outputs_dir, exist_ok=True)
    short_id = uuid.uuid4().hex[:6]
    run_output_dir = os.path.join(base_outputs_dir, short_id)
    os.makedirs(run_output_dir, exist_ok=True)

    logger.info("Invoking graph")
    retrieval_result = graph.invoke(
        {
            "original_query": "Make a hat in OpenSCAD",
            "messages": [],
            "vectors": vector_store,
            "bm25": bm25_retriever,
            "completed_assignments": [],
            "parts": [],
            "run_output_dir": run_output_dir,
        }
    )

    # Print detailed conversation trace
    print_conversation_trace(retrieval_result)


if __name__ == "__main__":
    main()
