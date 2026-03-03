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
    # Vault commands
    parser.add_argument("--vault-migrate", action="store_true",
                        help="Encrypt existing keys.json into vault and exit")
    parser.add_argument("--vault-set", nargs=2, metavar=("PROVIDER", "KEY"),
                        help="Set an API key in the vault and exit")
    parser.add_argument("--vault-list", action="store_true",
                        help="List providers stored in the vault and exit")
    # Backup commands
    parser.add_argument("--backup", action="store_true",
                        help="Create a backup and exit")
    parser.add_argument("--restore", type=str, metavar="FILE",
                        help="Restore from a backup file and exit")
    parser.add_argument("--list-backups", action="store_true",
                        help="List available backups and exit")
    args = parser.parse_args()

    # ── Vault CLI commands (exit early) ──
    if args.vault_migrate:
        from .vault import migrate_to_vault
        migrate_to_vault()
        print("✅ keys.json encrypted into vault")
        return
    if args.vault_set:
        from .vault import vault_set, vault_enabled
        if not vault_enabled():
            print("❌ Set LITEAGENT_VAULT_KEY environment variable first")
            sys.exit(1)
        vault_set(args.vault_set[0], args.vault_set[1])
        print(f"✅ {args.vault_set[0]} key saved to vault")
        return
    if args.vault_list:
        from .vault import vault_list, vault_enabled
        if not vault_enabled():
            print("❌ Set LITEAGENT_VAULT_KEY environment variable first")
            sys.exit(1)
        providers = vault_list()
        if providers:
            print("Keys in vault:")
            for p in providers:
                print(f"  • {p}")
        else:
            print("Vault is empty")
        return

    # ── Backup CLI commands (exit early) ──
    if args.list_backups:
        from .backup import list_backups
        backups = list_backups()
        if backups:
            for b in backups:
                print(f"  {b['name']}  ({b['size_kb']} KB)  {b['created_at']}")
        else:
            print("No backups found")
        return

    # Load config
    config = load_config(args.config)

    if args.backup:
        from .backup import backup
        path = backup(config.get("_config_path"))
        print(f"✅ Backup created: {path}")
        return
    if args.restore:
        from .backup import restore
        restore(args.restore)
        print("✅ Restored from backup. Restart liteagent to apply.")
        return

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

    # Determine channel — default is always "api" (dashboard)
    # Telegram runs alongside API automatically if enabled in config
    channel = args.channel or "api"

    # Setup scheduler
    from .scheduler import setup_scheduler
    scheduler = setup_scheduler(agent, config)
    if scheduler:
        agent._scheduler = scheduler  # Expose for dashboard API

    # Setup user tasks (persistent scheduled tasks)
    from .tasks import TaskManager, setup_task_checker
    task_manager = TaskManager(agent.memory.db)
    agent.enable_tasks(task_manager)
    if scheduler:
        setup_task_checker(scheduler, agent, task_manager)
    else:
        # Tasks need a scheduler — create a minimal one
        from .scheduler import Scheduler
        scheduler = Scheduler()
        setup_task_checker(scheduler, agent, task_manager)
        agent._scheduler = scheduler

    # Health monitor (exposed for dashboard + scheduler integration)
    from .health import HealthMonitor
    health_monitor = HealthMonitor(config)
    agent._health_monitor = health_monitor

    # Register health check as scheduler job (if scheduler available)
    if scheduler:
        health_cfg = config.get("health", {})
        if health_cfg.get("enabled", True):
            async def _health_check_job():
                results = await health_monitor.run_all_checks()
                for name, health in results.items():
                    if health.status == "down":
                        import logging
                        logging.getLogger("liteagent.health").warning(
                            "Channel %s is DOWN: %s", name, health.error_message)
            scheduler.add_job(
                "_health_check",
                health_cfg.get("cron", "*/5 * * * *"),
                _health_check_job,
                max_runtime_sec=30)

    try:
        if args.one_shot:
            response = asyncio.run(agent.run(args.one_shot, args.user))
            print(response)
        elif channel == "telegram":
            from .config import get_api_key
            tg_cfg = config.get("channels", {}).get("telegram", {})
            if not tg_cfg.get("token"):
                tg_cfg["token"] = get_api_key("telegram") or ""
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
                host = config.get("channels", {}).get("api", {}).get("host", "127.0.0.1")
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
            api_cfg = config.get("channels", {}).get("api", {})
            tg_cfg = config.get("channels", {}).get("telegram", {})

            # If Telegram is also enabled, start polling alongside API
            if tg_cfg.get("enabled") and tg_cfg.get("mode", "polling") == "polling":
                if not tg_cfg.get("token"):
                    from .config import get_api_key
                    tg_cfg["token"] = get_api_key("telegram") or ""

                async def _run_api_with_telegram():
                    """Run API server first, then start Telegram bot via HTTP proxy."""
                    import uvicorn
                    from .channels.api import create_app
                    from .channels.telegram import (
                        TelegramAPIClient, _parse_chat_ids,
                        _register_all_handlers, _set_bot_commands,
                    )

                    # Start scheduler
                    if scheduler:
                        scheduler.start()

                    # Start config watcher
                    config_path = config.get("_config_path")
                    watcher = None
                    if config_path:
                        from .config_watcher import ConfigWatcher
                        watcher = ConfigWatcher(config_path, agent, scheduler)
                        watcher.start()

                    # Create and start API server FIRST
                    app = create_app(agent)
                    host = api_cfg.get("host", "127.0.0.1")
                    port = api_cfg.get("port", 8080)

                    uvi_config = uvicorn.Config(app, host=host, port=port)
                    server = uvicorn.Server(uvi_config)
                    server_task = asyncio.create_task(server.serve())

                    # Wait for API to be ready
                    api_host = "127.0.0.1" if host == "0.0.0.0" else host
                    internal_token = getattr(app.state, 'internal_token', '')
                    api_client = TelegramAPIClient(
                        f"http://{api_host}:{port}", internal_token)

                    for _ in range(30):
                        if await api_client.health_check():
                            break
                        await asyncio.sleep(0.5)
                    else:
                        print("[ERROR] API server failed to start within 15s", flush=True)

                    # Auto-open dashboard in browser
                    import webbrowser
                    import threading
                    open_host = "localhost" if host == "0.0.0.0" else host
                    dashboard_url = f"http://{open_host}:{port}"
                    print(f"\n   Dashboard: {dashboard_url}\n")
                    threading.Timer(1.5, webbrowser.open, args=[dashboard_url]).start()

                    # Boot checks (proactive startup tasks)
                    from .boot import run_boot_checks
                    try:
                        boot_results = await run_boot_checks(agent, config)
                        for r in boot_results:
                            print(f"   Boot: {r['instruction']} → {r['status']}")
                    except Exception as e:
                        import logging
                        logging.getLogger("liteagent.boot").warning(
                            "Boot checks failed: %s", e)

                    # Start Telegram bot via HTTP proxy to API
                    tg_app = None
                    try:
                        from telegram.ext import Application
                        token = tg_cfg.get("token")
                        print(f"[Jess] Telegram token present: {bool(token)}", flush=True)
                        if token:
                            allowed_chat_ids = _parse_chat_ids(tg_cfg)
                            tg_app = Application.builder().token(token).build()
                            _register_all_handlers(
                                tg_app, api_client, allowed_chat_ids, tg_cfg)

                            await tg_app.initialize()
                            await tg_app.updater.start_polling(drop_pending_updates=True)
                            await tg_app.start()
                            agent._telegram_app = tg_app

                            has_rag = hasattr(agent, '_rag') and agent._rag is not None
                            await _set_bot_commands(tg_app.bot, has_rag=has_rag)

                            print("[Jess] Telegram bot running via API proxy!")
                    except Exception as e:
                        import traceback, sys
                        print(f"[ERROR] Telegram bot failed: {e}", flush=True)
                        traceback.print_exc(file=sys.stdout)
                        sys.stdout.flush()

                    # Wait for server to complete (blocks until shutdown)
                    try:
                        await server_task
                    finally:
                        await api_client.close()
                        if watcher:
                            watcher.stop()
                        if tg_app:
                            await tg_app.updater.stop()
                            await tg_app.stop()
                            await tg_app.shutdown()

                asyncio.run(_run_api_with_telegram())
            else:
                run_api_with_scheduler(agent, api_cfg, scheduler, full_config=config)
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
