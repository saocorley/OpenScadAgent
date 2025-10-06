import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from render_node import render_node

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class State(Dict):
    original_query: str
    code: str  # will always have the latest code.
    parts: List[Dict[str, str]]  # parts with respective colours.
    last_rendered_image: Any  # The last rendered image of the refiner agent.
    rendered_image_paths: List[str]  # All rendered images for this run
    saved_scad_path: str  # Path where current SCAD code was saved
    messages: List[BaseMessage]  # The messages of the refiner agent.
    feedback_tasks: List[str]  # The feedback tasks of the refiner agent.
    completed_feedback_tasks: List[
        str
    ]  # The completed feedback tasks of the refiner agent.
    run_output_dir: str  # Persist output directory across nodes
    rendered_output_dir: str  # Directory for rendered images


def build_graph():
    logger.info("Building refiner graph (nodes and edges)")
    graph = StateGraph(State)
    graph.add_node("render_node", render_node)
    graph.add_edge(START, "render_node")
    graph.add_edge("render_node", END)
    compiled = graph.compile()
    logger.info("Refiner graph compiled")
    return compiled


def main():
    load_dotenv()
    logger.info("Environment loaded for refiner agent")
    # Hardcoded path to the final model for quick testing.
    # SCAD_PATH = "/home/scorley/OpenScadAgent/output/models/assembled_session_20250912_142239/generated_model_20250912_142239.scad"
    SCAD_PATH = "/home/scorley/OpenScadAgent/outputs/coder_outputs/f06f0f/assembled_session_20250912_175638/generated_model_20250912_175638.scad"
    code = ""
    if SCAD_PATH and os.path.exists(SCAD_PATH):
        with open(SCAD_PATH, "r") as file:
            code = file.read()
        logger.info("Using SCAD source: %s (chars=%d)", SCAD_PATH, len(code))
    else:
        logger.info("No SCAD source found at %s; starting with empty code", SCAD_PATH)
    # Create per-run output directory for the refiner agent
    import uuid

    base_outputs_dir = "/home/scorley/OpenScadAgent/outputs/refiner_outputs"
    os.makedirs(base_outputs_dir, exist_ok=True)
    short_id = uuid.uuid4().hex[:6]
    run_output_dir = os.path.join(base_outputs_dir, short_id)
    os.makedirs(run_output_dir, exist_ok=True)
    rendered_output_dir = os.path.join(run_output_dir, "rendered_images")
    os.makedirs(rendered_output_dir, exist_ok=True)
    logger.info(
        "Refiner run ID: %s | run_output_dir=%s | rendered_output_dir=%s",
        short_id,
        run_output_dir,
        rendered_output_dir,
    )

    graph = build_graph()
    initial_state = {
        "original_query": "Make me a rocketship in OpenSCAD",
        "code": code,
        "parts": [],
        "messages": [],
        "feedback_tasks": [],
        "completed_feedback_tasks": [],
        "run_output_dir": run_output_dir,
        "rendered_output_dir": rendered_output_dir,
    }
    logger.info(
        "Invoking refiner graph | code_chars=%d | run_output_dir=%s",
        len(code),
        run_output_dir,
    )
    final_state = graph.invoke(initial_state)
    try:
        rendered = (
            final_state.get("rendered_image_paths", [])
            if isinstance(final_state, dict)
            else []
        )
        logger.info(
            "Refiner graph complete | rendered_images=%d%s",
            len(rendered),
            f" | last={rendered[-1]}" if rendered else "",
        )
    except Exception:
        # Best-effort logging; do not break on logging issues
        pass


if __name__ == "__main__":
    main()
