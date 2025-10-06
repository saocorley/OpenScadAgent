## OpenScadAgent

This repo contains a multi‑agent system that plans, codes, and assembles parameterized OpenSCAD models. It uses a small team of specialized agents wired together with a state graph to take a natural‑language request and produce OpenSCAD code (and optionally renders).

Key technologies: LangGraph, LangChain, OpenAI models, BM25/In‑memory vector store, OpenSCAD CLI.


### Architecture at a glance

- **Planner Agent** (`agents/open_scad_generator_graph/planner_agent.py` with prompt in `agents/prompts/planner_agent.txt`)
  - Breaks a user request into parts and assigns work to other agents.
  - Always schedules a retrieval step immediately before coding or assembly.
  - Outputs a list of assignments like ["retrieval agent", "coding agent", "assembly agent"].

- **Retrieval Agent** (`agents/open_scad_generator_graph/retrieval_agent.py` with prompt in `agents/prompts/retrieval_agent.txt`)
  - Returns concise, task‑specific references from OpenSCAD docs.
  - Backed by a simple vector store and BM25 retriever prepared from curated wiki text files.

- **Coding Agent** (`agents/open_scad_generator_graph/coding_agent.py` with prompt in `agents/prompts/coding_agent.txt`)
  - Produces exactly one parameterized OpenSCAD module for a single part.
  - Saves the generated `.scad` under a per‑run folder and optionally runs `sca2d` to check syntax.
  - No top‑level geometry; only `module <name>(...) { ... }`.

- **Assembly Agent** (`agents/open_scad_generator_graph/assembly_agent.py` with prompt in `agents/prompts/assembly_agent.txt`)
  - Composes the final model by calling the part modules with transforms/CSG.
  - Applies distinct `color()` to each part to aid visualization.
  - Does not rewrite part internals; it only positions and combines them.

- **Graph Orchestration** (`agents/open_scad_generator_graph/main.py`)
  - Builds a LangGraph `StateGraph` with nodes: planner → retrieval/coding/assembly.
  - Prepares retrievers from `bibliography/wiki_*.txt` and creates a per‑run output dir under `outputs/coder_outputs/<short_id>`.

- **Refiner Agent (experimental / in development)** (`agents/refiner_agent/render_node.py` and prompt in `agents/prompts/render_agent.txt`)
  - Uses the OpenSCAD CLI to render images from a given `.scad` file and sensible camera views.
  - Status: still being integrated and stabilized; interface and behavior may change.


### How the OpenSCAD agent works (end‑to‑end)

1) You provide a high‑level request (e.g., “Make a hat in OpenSCAD”).
2) The Planner produces a plan with assignments and routes work to:
   - Retrieval for focused documentation snippets
   - Coding to generate a single, parameterized part module
   - Assembly to compose multiple parts into a final model
3) The Coding Agent writes the part module and saves it under the current run’s folder.
4) Optionally, the Refiner (WIP) renders images via the OpenSCAD CLI for quick inspection.


### Running locally

Prerequisites:
- Python 3.10+
- OpenAI API access
- OpenSCAD installed and available on PATH (for rendering; required by Refiner; optional otherwise)
- Optional: `sca2d` if you want the coding agent’s quick syntax checks

Environment:
- Create a local `.env` file (not tracked) with your key:
  - `OPENAI_API_KEY=...`

Install dependencies (example):
```bash
pip install langchain langchain-core langchain-community langchain-openai langgraph python-dotenv
```

Run the sample graph:
```bash
python -m agents.open_scad_generator_graph.main
```

Outputs:
- Generated parts are written under `outputs/coder_outputs/<short_id>/`.
- If you enable/refactor rendering, images will be placed under a corresponding outputs directory (see `agents/refiner_agent/render_node.py`).


### Repository layout (selected)

- `agents/open_scad_generator_graph/` — planner, retrieval, coding, assembly, and the LangGraph wiring (`main.py`).
- `agents/refiner_agent/` — experimental renderer node that calls OpenSCAD.
- `agents/prompts/` — system prompts for each agent.
- `bibliography/` — curated OpenSCAD wiki text used for retrieval.
- `outputs/` — run‑specific artifacts.


### Development status: Refiner Agent

The Refiner agent is still in development and not yet considered stable. Expect changes in its interface, view selection, and output paths as integration improves.


### Security and secrets

- Secrets belong in local `.env` files only; they are ignored by `.gitignore`.
- If a key is ever committed by mistake, rotate it immediately in your provider’s dashboard and rewrite history before pushing.


