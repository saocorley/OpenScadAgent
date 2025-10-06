import json
import logging
import os
from typing import Dict, List

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .coding_agent import compile_code

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class AssemblyPart(BaseModel):
    name: str
    code: str
    color: str


class AssemblyResult(BaseModel):
    final_code: str
    parts: List[AssemblyPart]
    colours: Dict[str, str]


PALETTE: List[str] = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "lime",
    "pink",
    "teal",
    "brown",
    "gray",
]

# Load system prompt once at module level
with open("agents/prompts/assembly_agent.txt", "r") as file:
    ASSEMBLY_SYSTEM_PROMPT = file.read()

# Create LLM object and structured output wrapper
ASSEMBLY_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ASSEMBLY_STRUCTURED_LLM = ASSEMBLY_LLM.with_structured_output(
    AssemblyResult, method="function_calling"
)


def assembly_agent(state) -> dict:
    logger.info("Starting LLM-based assembly agent...")

    parts_raw: List[Dict[str, str]] = state.get("parts", [])
    if not parts_raw:
        logger.warning("No parts found in state; nothing to assemble")
        state["assembly_result"] = AssemblyResult(
            final_code="", parts=[], colours={}
        ).model_dump()
        # Mark assignment complete if present
        if state.get("assignments"):
            completed_assignment = state["assignments"].pop(0)
            state.setdefault("completed_assignments", []).append(completed_assignment)
        return state

    # If any color is missing, assign from palette for better visualization hints
    enriched_parts: List[Dict[str, str]] = []
    for index, part in enumerate(parts_raw):
        if "color" not in part or not part["color"]:
            part = {**part, "color": PALETTE[index % len(PALETTE)]}
        enriched_parts.append(part)

    plan_text = state.get("plan", "")
    context_text = state.get("lastest_context", "")

    sys_message = SystemMessage(content=ASSEMBLY_SYSTEM_PROMPT)
    human_payload = {
        "original_query": state.get("original_query", ""),
        "plan": plan_text,
        "parts": enriched_parts,
        "context": context_text,
    }
    human_message = HumanMessage(content=json.dumps(human_payload))

    logger.info("Invoking assembly LLM with %d parts", len(enriched_parts))
    response: AssemblyResult = ASSEMBLY_STRUCTURED_LLM.invoke(
        [sys_message, human_message]
    )

    # Persist to state as dicts
    state["assembly_result"] = response.model_dump()

    # Compile and render the final assembly
    # IMPORTANT: Use the original part modules unchanged; append assembly logic after.
    original_parts = state.get("parts", [])
    parts_code = "\n".join([p.get("code", "") for p in original_parts])

    # Optional uniform scaling wrapper for the assembly to improve render fit
    scale_from_state = state.get("assembly_scale")
    env_scale = os.getenv("OPENSCAD_ASSEMBLY_SCALE")
    try:
        scale_factor = float(
            scale_from_state
            if scale_from_state is not None
            else (env_scale if env_scale is not None else 1.0)
        )
    except (TypeError, ValueError):
        scale_factor = 1.0

    if scale_factor != 1.0:
        assembly_code = f"scale([{scale_factor},{scale_factor},{scale_factor}]){{\n{response.final_code}\n}}"
    else:
        assembly_code = response.final_code


    run_output_dir = state.get("run_output_dir")
    if not run_output_dir:
        logger.warning(
            "run_output_dir missing from state; creating per-run dir under outputs/coder_outputs"
        )
        import uuid

        base_outputs_dir = "/home/scorley/OpenScadAgent/outputs/coder_outputs"
        os.makedirs(base_outputs_dir, exist_ok=True)
        short_id = uuid.uuid4().hex[:6]
        run_output_dir = os.path.join(base_outputs_dir, short_id)
        os.makedirs(run_output_dir, exist_ok=True)
        state["run_output_dir"] = run_output_dir
    compile_result = compile_code(
        parts_code + "\n" + assembly_code,
        run_output_dir=run_output_dir,
        kind="assembled",
    )
    state["assembly_compile_result"] = compile_result
    if "error" not in compile_result and compile_result.get("return_code") == 0:
        logger.info("Final assembly compiled successfully; rendering images...")

    summary = "LLM Assembly complete. Parts: " + ", ".join(
        [f"{p.name}({p.color})" for p in response.parts]
    )
    state.setdefault("messages", []).append(AIMessage(content=summary))

    # Mark assignment complete
    if state.get("assignments"):
        completed_assignment = state["assignments"].pop(0)
        state.setdefault("completed_assignments", []).append(completed_assignment)

    return state
