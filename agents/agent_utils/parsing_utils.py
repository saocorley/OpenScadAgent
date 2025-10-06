import re

def router(planner_output: str):
    """
    Extracts only <agent>:<task> lines from planner output
    and returns a list of (agent, task) tuples.
    """
    tasks = []
    # Only match retrieval agent: or coding agent: at the start of a line
    pattern = re.compile(r"^(retrieval agent|coding agent)\s*:\s*(.*)$", re.IGNORECASE)

    for line in planner_output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue  # ignore reasoning or other lines
        agent, task = match.groups()
        tasks.append((agent.lower(), task.strip()))


    return tasks

def parse_retrieval_output(query: str, tool_messages):
    """
    Parses the retrieval output into a message that contains the context
    """
    if not tool_messages:
        return "No retrieval results found"
    
    # Combine all tool message contents
    combined_context = ""
    for tool_message in tool_messages:
        combined_context += tool_message.content + "\n\n"
    
    return f"Retrieved information for query '{query}':\n\n{combined_context.strip()}"