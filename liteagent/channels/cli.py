"""CLI channel — interactive REPL for LiteAgent."""

import asyncio
import sys

from ..agent import LiteAgent


BANNER = """
╔══════════════════════════════════════╗
║  🤖 LiteAgent v1.0.0                ║
║  Type /help for commands, Ctrl+C exit║
╚══════════════════════════════════════╝
"""


async def run_cli(agent: LiteAgent, user_id: str = "cli-user"):
    """Run interactive CLI loop."""
    print(BANNER)

    while True:
        try:
            # Read input
            try:
                user_input = input("\n\033[36myou>\033[0m ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                print("\n👋 Bye!")
                break

            # Check for commands
            cmd_response = agent.handle_command(user_input, user_id)
            if cmd_response is not None:
                print(f"\n\033[33m{cmd_response}\033[0m")
                continue

            # Run agent with streaming
            print("\n\033[32magent>\033[0m ", end="", flush=True)
            async for chunk in agent.stream(user_input, user_id):
                print(chunk, end="", flush=True)
            print()  # Newline after stream ends

        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n\033[31m❌ Error: {e}\033[0m")
