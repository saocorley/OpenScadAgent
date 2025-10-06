import logging
import os
import subprocess
import uuid  # For unique filenames

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

# Load environment so OPENAI_API_KEY is available at import time
dotenv.load_dotenv()

# Assuming logger is configured elsewhere
logger = logging.getLogger(__name__)

with open("agents/prompts/render_agent.txt", "r") as file:
    RENDER_SYSTEM_PROMPT = file.read()


def _render_images(
    filepath: str,
    views: dict[str, str] | None = None,
    img_width: int = 1024,
    img_height: int = 1024,
    orthographic: bool = True,
    use_autocenter: bool = True,
    use_viewall: bool = True,
    render_scale: float | None = None,
) -> list[str]:
    """
    Renders multiple images of an OpenSCAD model and returns their file paths.

    Args:
        filepath: Path to the .scad file.
        views: Optional dict of view_name -> "[pitch,yaw,roll]" string.
        img_width: Output image width in pixels.
        img_height: Output image height in pixels.
        orthographic: If True, use orthographic projection.
        use_autocenter: If True, pass --autocenter.
        use_viewall: If True, auto-fit the model in the view.
        render_scale: If not 1.0, exports and renders a scaled STL for better framing.

    Returns:
        A list of file paths for the successfully rendered images.
    """
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}")
        return []

    output_dir = os.path.dirname(filepath)
    # Generate a unique base name for this render job's files
    job_id = uuid.uuid4().hex[:8]
    base_filename = f"{os.path.basename(filepath).replace('.scad', '')}_{job_id}"

    source_to_render = filepath

    try:
        # --- Handle non-destructive scaling ---
        if render_scale and abs(render_scale - 1.0) > 1e-9:
            stl_path = os.path.join(output_dir, f"{base_filename}_temp.stl")
            wrapper_path = os.path.join(output_dir, f"{base_filename}_wrapper.scad")

            stl_cmd = ["openscad", filepath, "-o", stl_path]
            subprocess.run(stl_cmd, capture_output=True, text=True, check=True)

            wrapper_code = f'scale([{render_scale},{render_scale},{render_scale}]) import("{stl_path}");'
            with open(wrapper_path, "w") as wf:
                wf.write(wrapper_code)

            source_to_render = wrapper_path
            logger.info(f"Using render-only scaled wrapper (scale={render_scale})")

        # --- Define default views if none provided ---
        if views is None:
            iso_pitch = 35.264
            views = {
                "iso": f"[{iso_pitch},0,45]",
                "top": "[90,0,0]",
                "front": "[0,0,0]",
            }

        # --- Loop through views and render ---
        rendered_image_paths = []
        for view_name, rotation in views.items():
            output_path = os.path.join(output_dir, f"{base_filename}_{view_name}.png")
            projection = "o" if orthographic else "p"

            cmd = [
                "openscad",
                source_to_render,
                "-o",
                output_path,
                f"--imgsize={img_width},{img_height}",
                f"--projection={projection}",
                "-D",
                f"$vpr={rotation}",
            ]
            if use_autocenter:
                cmd.append("--autocenter")
            if use_viewall:
                cmd.append("--viewall")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Rendered {view_name} view to {output_path}")
                rendered_image_paths.append(output_path)
            else:
                logger.error(f"Failed to render {view_name}: {result.stderr}")

        return rendered_image_paths

    except subprocess.CalledProcessError as e:
        logger.error(
            f"STL export failed for scaling; cannot render. STDERR: {e.stderr}"
        )
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred in render_images: {e}")
        return []


def render_images(
    filepath: str,
    views: dict[str, str] | None = None,
    img_width: int = 1024,
    img_height: int = 1024,
    orthographic: bool = True,
    use_autocenter: bool = True,
    use_viewall: bool = True,
    render_scale: float | None = None,
) -> list[str]:
    """
    Tool wrapper: renders a provided SCAD filepath with optional view params.
    """
    return _render_images(
        filepath=filepath,
        views=views,
        img_width=img_width,
        img_height=img_height,
        orthographic=orthographic,
        use_autocenter=use_autocenter,
        use_viewall=use_viewall,
        render_scale=render_scale,
    )


tool = StructuredTool.from_function(
    name="render_images",
    description="Renders the OpenSCAD code and returns the rendered images.",
    func=render_images,
)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([tool])


def ensure_run_output_dir(state) -> str:
    """
    Ensure there is a run_output_dir in state. If missing, create a manual one
    under outputs/refiner_outputs and persist it back to state.
    """
    run_output_dir = state.get("run_output_dir")
    if not run_output_dir:
        run_output_dir = "/home/scorley/OpenScadAgent/outputs/refiner_outputs/manual"
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            state["run_output_dir"] = run_output_dir
        except Exception as e:
            logger.error(f"Failed to create manual run_output_dir: {e}")
    return run_output_dir


def save_scad_code_to_run_dir(code: str, run_output_dir: str) -> str:
    """
    Save the provided SCAD code into the run directory with a timestamped name
    and return the saved file path.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scad_filename = f"refiner_{timestamp}.scad"
    saved_scad_path = os.path.join(run_output_dir, scad_filename)
    with open(saved_scad_path, "w") as f:
        f.write(code or "")
    return saved_scad_path


def execute_render_tool_calls(response, filepath: str):
    """
    Given an LLM response with tool_calls, execute the render tool once using
    only the allowed arguments and return the list of rendered image paths.
    """
    if not response.tool_calls:
        return []

    for tool_call in response.tool_calls:
        logger.info(
            f"Executing tool: {tool_call['name']} with args: {tool_call['args']}"
        )
        provided_args = dict(tool_call.get("args") or {})
        allowed_keys = {
            "filepath",
            "views",
            "img_width",
            "img_height",
            "orthographic",
            "use_autocenter",
            "use_viewall",
            "render_scale",
        }
        filtered_args = {k: v for k, v in provided_args.items() if k in allowed_keys}
        # Ensure filepath is always provided to the tool (no globals)
        filtered_args["filepath"] = filepath
        return tool.invoke(filtered_args) or []

    return []


def move_rendered_images(rendered_image_paths, target_render_dir):
    """
    Move or copy rendered images into the target directory and return new paths.
    """
    if not target_render_dir or not rendered_image_paths:
        return rendered_image_paths or []

    os.makedirs(target_render_dir, exist_ok=True)
    moved_paths = []
    for src_path in rendered_image_paths:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(target_render_dir, filename)
        try:
            os.replace(src_path, dst_path)
        except Exception:
            import shutil

            shutil.copy2(src_path, dst_path)
            try:
                os.remove(src_path)
            except Exception:
                pass
        moved_paths.append(dst_path)
    return moved_paths


def render_node(state):
    """
    Renders the OpenSCAD code and returns the rendered images.
    """
    # Resolve output directory and save code
    run_output_dir = ensure_run_output_dir(state)
    try:
        saved_scad_path = save_scad_code_to_run_dir(
            state.get("code", ""), run_output_dir
        )
        state["saved_scad_path"] = saved_scad_path
        logger.info(f"Saved refiner .scad to: {saved_scad_path}")
    except Exception as e:
        logger.error(f"Failed to save .scad code: {e}")

    # Compose messages and invoke LLM with tool; filepath handled internally by wrapper
    sys_message = SystemMessage(content=RENDER_SYSTEM_PROMPT)
    messages = [
        sys_message,
        HumanMessage(content="Render the provided code using the appropriate views."),
    ]
    response = llm_with_tools.invoke(messages)
    logger.info("Tool calls detected: %s", len(response.tool_calls or []))
    rendered_image_paths = execute_render_tool_calls(response, saved_scad_path)

    # If a target render directory is provided in state, move files there
    target_render_dir = state.get("rendered_output_dir") or state.get("run_output_dir")
    try:
        rendered_image_paths = move_rendered_images(
            rendered_image_paths, target_render_dir
        )
    except Exception as e:
        logger.error(f"Failed to move rendered images: {e}")

    # Store paths in state
    if rendered_image_paths:
        state["rendered_image_paths"] = rendered_image_paths
        state["last_rendered_image"] = rendered_image_paths[-1]
        logger.info(f"Image rendered to {rendered_image_paths[-1]}")
    else:
        logger.warning("No images were rendered")
    return state
