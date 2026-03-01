"""LiteAgent — entry point."""

import argparse
import asyncio
import atexit
import signal
import sys

from .config import load_config
from .agent import LiteAgent
from .channels.cli import run_cli


def main():
    parser = argparse.ArgumentParser(description="LiteAgent — lightweight AI agent")
    parser.add_argument("-c", "--config", default=None, help="Path to config.json")
    parser.add_argument("--user", default="cli-user", help="User ID for memory isolation")
    parser.add_argument("--one-shot", "-1", default=None, help="Run single query and exit")
    parser.add_argument("--channel", default=None, choices=["cli", "telegram", "api"],
                        help="Channel to use (default: auto-detect from config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create agent(s) — use pool if multi-agent config present
    if "agents" in config:
        from .pool import AgentPool
        pool = AgentPool.from_config(config)
        agent = pool.default
        cleanup = pool.close_all
    else:
        agent = LiteAgent(config)
        pool = None
        cleanup = agent.memory.close

    # Ensure cleanup on any exit path
    atexit.register(cleanup)

    def _shutdown(signum, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    # Determine channel
    channel = args.channel
    if not channel:
        channels_cfg = config.get("channels", {})
        for name in ["telegram", "api", "cli"]:
            ch = channels_cfg.get(name, {})
            if isinstance(ch, dict) and ch.get("enabled"):
                channel = name
                break
        channel = channel or "cli"

    # Setup scheduler
    from .scheduler import setup_scheduler
    scheduler = setup_scheduler(agent, config)
    if scheduler:
        agent._scheduler = scheduler  # Expose for dashboard API

    try:
        if args.one_shot:
            response = asyncio.run(agent.run(args.one_shot, args.user))
            print(response)
        elif channel == "telegram":
            tg_cfg = config.get("channels", {}).get("telegram", {})
            mode = tg_cfg.get("mode", "polling")
            if mode == "webhook":
                # Webhook mode: run Telegram alongside API server
                from .channels.api import create_app
                from .channels.telegram import setup_webhook_route
                app = create_app(agent)
                setup_webhook_route(app, agent, tg_cfg)
                if scheduler:
                    @app.on_event("startup")
                    async def _start_scheduler():
                        scheduler.start()
                import uvicorn
                host = config.get("channels", {}).get("api", {}).get("host", "0.0.0.0")
                port = config.get("channels", {}).get("api", {}).get("port", 8080)
                uvicorn.run(app, host=host, port=port)
            else:
                from .channels.telegram import run_telegram

                async def _run_tg_with_scheduler():
                    if scheduler:
                        scheduler.start()
                    await run_telegram(agent, tg_cfg)
                asyncio.run(_run_tg_with_scheduler())
        elif channel == "api":
            from .channels.api import run_api_with_scheduler
            run_api_with_scheduler(agent, config.get("channels", {}).get("api", {}), scheduler)
        else:
            async def _run_cli_with_scheduler():
                if scheduler:
                    scheduler.start()
                await run_cli(agent, args.user)
            asyncio.run(_run_cli_with_scheduler())
    finally:
        if scheduler:
            scheduler.stop()
        cleanup()


if __name__ == "__main__":
    main()
