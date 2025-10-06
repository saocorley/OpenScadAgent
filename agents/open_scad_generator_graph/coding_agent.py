import logging
import os
import subprocess
from datetime import datetime

import dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Load system prompt once at module level
with open("agents/prompts/coding_agent.txt", "r") as file:
    CODING_SYSTEM_PROMPT = file.read()

# Create LLM object once at module level
CODING_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def compile_code(code: str, run_output_dir: str, kind: str):
    """
    Save OpenSCAD code under the per-run directory and run sca2d to check for errors.

    Args:
        code: The OpenSCAD code to compile and check
        run_output_dir: The base directory for this graph run (e.g., outputs/coder outputs/<short_id>)
        kind: Either "part" or "assembled" to prefix the session folder name

    Returns:
        dict: Contains the filename, error count, and full sca2d output
    """
    try:
        # Generate timestamp for unique folder naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Validate kind and build session folder under the run directory with prefix
        prefix = (
            "part"
            if kind == "part"
            else ("assembled" if kind == "assembled" else "unknown")
        )
        session_folder = os.path.join(run_output_dir, f"{prefix}_session_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)

        logger.info(f"Created session folder: {session_folder}")

        # Generate filename
        filename = f"generated_model_{timestamp}.scad"
        filepath = os.path.join(session_folder, filename)

        # Save the code to file
        with open(filepath, "w") as f:
            f.write(code)

        logger.info(f"Saved OpenSCAD code to: {filepath}")

        # Run sca2d command to check for errors
        try:
            result = subprocess.run(
                ["sca2d", filepath],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            # Print the result as requested

            if result.stderr:
                print(f"STDERR:\n{result.stderr}")

            # Parse error count from output if possible
            error_count = "Unknown"
            if result.stdout:
                # Look for common error indicators in the output
                lines = result.stdout.split("\n")
                for line in lines:
                    if "error" in line.lower() or "ERROR" in line:
                        print(f"Found error line: {line}")

            return {
                "filename": filename,
                "filepath": filepath,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error_count": error_count,
                "session_folder": session_folder,
            }

        except subprocess.TimeoutExpired:
            error_msg = "sca2d command timed out after 30 seconds"
            logger.error(error_msg)
            print(error_msg)
            return {
                "filename": filename,
                "filepath": filepath,
                "session_folder": session_folder,
                "error": error_msg,
            }

        except FileNotFoundError:
            error_msg = "sca2d command not found. Please make sure it's installed and in your PATH."
            logger.error(error_msg)
            print(error_msg)
            return {
                "filename": filename,
                "filepath": filepath,
                "session_folder": session_folder,
                "error": error_msg,
            }

    except Exception as e:
        error_msg = f"Error in compile_code: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return {"error": error_msg}


def coding_agent(state) -> dict:
    """This agent is responsible for coding the OpenSCAD model."""
    logger.info("Starting coding agent...")

    sys_message = SystemMessage(content=CODING_SYSTEM_PROMPT)
    task = state["assignments"][0][1]
    context = state.get("lastest_context", "There is no context avaible for this task.")
    messages = (
        [sys_message]
        + [
            AIMessage(
                content="This is your task, as stated by the planner agent: \n" + task
            )
        ]
        + [
            AIMessage(
                content=f"""We have scanned the official OpenSCAD documentation and found the following information, which might be useful to code the OpenSCAD model: \n
                                            {context}\n"""
            )
        ]
    )
    logger.info("Generating OpenSCAD code...")
    response = CODING_LLM.invoke(messages)
    code = response.content
    logger.info("OpenSCAD code generated successfully")

    # Use compile_code to save and check the OpenSCAD code (under run directory as a part)
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
    compile_result = compile_code(code, run_output_dir=run_output_dir, kind="part")

    # Check if code compilation was successful and render images if no errors

    if "error" in compile_result:
        logger.error(f"Failed to compile code: {compile_result['error']}")

    # Extract module name for part naming (strict; no fallback)
    import re

    module_match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    if not module_match:
        logger.error("No module definition found in generated code.")
        state["compile_result"] = compile_result
        state["messages"].append(
            AIMessage(
                content=(
                    "Coding agent error: expected a single module definition but none was found."
                )
            )
        )
        return state
    part_name = module_match.group(1)

    # Add the compile result to the state for further processing
    state["compile_result"] = compile_result
    # Store the part for later assembly
    state.setdefault("parts", []).append({"name": part_name, "code": code})
    state["completed_assignments"].append(("coding agent", task))
    state["assignments"].pop(0)
    state["messages"].append(
        AIMessage(
            content=f"Code resulting from the coding agent and query {task}: " + code
        )
    )
    return state
