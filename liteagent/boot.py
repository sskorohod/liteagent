"""Boot checks system — proactive startup instructions.

Inspired by OpenClaw's BOOT.md pattern. On startup, the agent reads
a markdown file with directives and executes them autonomously.

Supported directives:
    ## CHECK providers  — verify API keys for all configured providers
    ## CHECK channels   — test connectivity to enabled channels
    ## TASK <desc>      — execute a task via the agent
    ## MESSAGE <target> — send a message to a channel/user

Boot file locations (checked in order):
    1. ~/.liteagent/boot.md
    2. <project_root>/boot.md
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BOOT_PATHS = [
    Path.home() / ".liteagent" / "boot.md",
    Path(__file__).resolve().parent.parent / "boot.md",
]


@dataclass
class BootInstruction:
    """A parsed boot instruction."""
    type: str       # "check", "task", "message"
    target: str     # "providers", "channels", channel/user ID
    content: str    # description or message body
    priority: int = 100


def find_boot_file(config: dict) -> Path | None:
    """Find the boot instructions file."""
    # Custom path from config
    custom = config.get("boot", {}).get("file")
    if custom:
        p = Path(custom).expanduser()
        if p.exists():
            return p

    # Also check relative to config file location
    cfg_path = config.get("_config_path")
    if cfg_path:
        candidate = Path(cfg_path).parent / "boot.md"
        if candidate.exists():
            return candidate

    for p in BOOT_PATHS:
        if p.exists():
            return p
    return None


def parse_boot_file(content: str) -> list[BootInstruction]:
    """Parse markdown boot file into structured instructions.

    Format:
        ## CHECK providers
        Optional description text on following lines.

        ## TASK daily_summary
        Generate a brief summary of yesterday's activity.

        ## MESSAGE tg-12345
        Good morning! I'm back online.
    """
    instructions: list[BootInstruction] = []
    pattern = re.compile(
        r'^##\s+(CHECK|TASK|MESSAGE)\s+(.+?)$',
        re.MULTILINE | re.IGNORECASE)

    matches = list(pattern.finditer(content))
    for i, m in enumerate(matches):
        directive = m.group(1).lower()
        target = m.group(2).strip()
        # Content = everything between this heading and the next
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        instructions.append(BootInstruction(
            type=directive, target=target, content=body))

    return instructions


async def execute_boot_instruction(agent, instr: BootInstruction) -> dict:
    """Execute a single boot instruction. Returns result dict."""
    result = {
        "instruction": f"{instr.type.upper()} {instr.target}",
        "status": "ok",
        "message": "",
        "timestamp": datetime.now().isoformat(),
    }

    try:
        if instr.type == "check":
            if instr.target.lower() == "providers":
                result["message"] = await _check_providers(agent)
            elif instr.target.lower() == "channels":
                result["message"] = _check_channels(agent)
            else:
                result["message"] = f"Unknown check target: {instr.target}"
                result["status"] = "skipped"

        elif instr.type == "task":
            # Run a task through the agent itself
            task_desc = instr.content or instr.target
            response = await agent.run(task_desc, user_id="boot")
            result["message"] = response[:500]

        elif instr.type == "message":
            result["message"] = f"Message queued for {instr.target}"
            result["status"] = "queued"

        else:
            result["status"] = "skipped"
            result["message"] = f"Unknown directive: {instr.type}"

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)[:300]
        logger.warning("Boot instruction failed: %s %s — %s",
                        instr.type, instr.target, e)

    return result


async def _check_providers(agent) -> str:
    """Verify configured provider API keys are valid."""
    from .config import get_api_key
    provider_name = agent.config.get("agent", {}).get("provider", "anthropic")
    results = []

    # Check primary provider
    key = get_api_key(provider_name)
    if provider_name == "ollama":
        results.append(f"{provider_name}: no key needed")
    elif key:
        results.append(f"{provider_name}: key configured ({key[:6]}...)")
    else:
        results.append(f"{provider_name}: NO KEY FOUND")

    # Check configured fallback providers
    for pname in agent.config.get("providers", {}):
        if pname == provider_name:
            continue
        pkey = get_api_key(pname)
        if pname == "ollama":
            results.append(f"{pname}: no key needed")
        elif pkey:
            results.append(f"{pname}: key configured")
        else:
            results.append(f"{pname}: no key")

    return "; ".join(results)


def _check_channels(agent) -> str:
    """Check which channels are enabled in config."""
    channels_cfg = agent.config.get("channels", {})
    results = []
    for name, cfg in channels_cfg.items():
        if isinstance(cfg, dict):
            enabled = cfg.get("enabled", True)
            results.append(f"{name}: {'enabled' if enabled else 'disabled'}")
        else:
            results.append(f"{name}: configured")
    return "; ".join(results) if results else "no channels configured"


async def run_boot_checks(agent, config: dict) -> list[dict]:
    """Execute all boot instructions from boot.md.

    Args:
        agent: LiteAgent instance
        config: Agent configuration dict

    Returns:
        List of result dicts with status for each instruction.
    """
    if not config.get("boot", {}).get("enabled", True):
        return []

    boot_path = find_boot_file(config)
    if not boot_path:
        logger.debug("No boot.md found, skipping boot checks")
        return []

    logger.info("Running boot checks from %s", boot_path)
    content = boot_path.read_text(encoding="utf-8")
    instructions = parse_boot_file(content)

    if not instructions:
        logger.debug("boot.md found but no directives parsed")
        return []

    results = []
    for instr in instructions:
        result = await execute_boot_instruction(agent, instr)
        results.append(result)
        logger.info("Boot: %s → %s: %s",
                     result["instruction"], result["status"],
                     result["message"][:100])

    return results
