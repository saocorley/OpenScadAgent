import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


def print_conversation_trace(state):
    """
    Print all messages in state['messages'] in a clear, readable format for debugging
    """
    print("\n" + "=" * 80)
    print("CONVERSATION TRACE")
    print("=" * 80)

    messages = state.get("messages", [])

    if not messages:
        print("No messages found in state")
        return

    for i, message in enumerate(messages):
        print(f"\n--- MESSAGE {i + 1} ---")

        if isinstance(message, SystemMessage):
            print("🔧 SYSTEM MESSAGE")
            print(f"Content: {message.content[:200]}...")

        elif isinstance(message, HumanMessage):
            print("👤 HUMAN MESSAGE")
            print(f"Content: {message.content}")

        elif isinstance(message, AIMessage):
            print("🤖 AI MESSAGE")
            print(f"Content: {message.content}")

            # Show tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"🔧 Tool Calls: {len(message.tool_calls)}")
                for j, tool_call in enumerate(message.tool_calls):
                    print(f"  Tool {j + 1}: {tool_call.get('name', 'unknown')}")
                    print(f"  Args: {tool_call.get('args', {})}")

        elif isinstance(message, ToolMessage):
            print("🛠️ TOOL MESSAGE")
            print(f"Tool Call ID: {message.tool_call_id}")
            # tool messages are cut cause they be long.
            print(f"Content Preview: {str(message.content[:100])}...")

        else:
            print(f"❓ UNKNOWN MESSAGE TYPE: {type(message)}")
            print(f"Content: {str(message)}")

    # Show final state summary
    print("\n--- FINAL STATE SUMMARY ---")
    print(f"Total Messages: {len(messages)}")
    print(f"Plan Text: {state.get('plan', 'Not found')}")
    print(f"Assignments: {state.get('assignments', 'Not found')}")
    print("=" * 80 + "\n")
