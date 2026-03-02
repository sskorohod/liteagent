"""Plugin loader — discover and load plugins from ~/.liteagent/plugins/.

Each plugin is a .py file with a register(hooks, config) function.
Plugins register hook handlers to extend agent behavior.

Example plugin (~/.liteagent/plugins/my_plugin.py):

    def register(hooks, config):
        async def on_response(ctx):
            # Modify response text, log metrics, etc.
            pass
        hooks.register("after_response", "my_plugin_hook", on_response, priority=300, plugin="my_plugin")
"""

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGINS_DIR = Path.home() / ".liteagent" / "plugins"


def load_plugins(hooks, config: dict) -> list[str]:
    """Load Python plugins from plugins directory.

    Each plugin must export a register(hooks, config) function.

    Args:
        hooks: HookRegistry instance
        config: Agent configuration dict

    Returns:
        List of loaded plugin names.
    """
    if not PLUGINS_DIR.exists():
        return []

    loaded = []
    for py_file in sorted(PLUGINS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(name, py_file)
            if spec is None or spec.loader is None:
                logger.warning("Plugin %s: failed to create module spec", name)
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                mod.register(hooks, config)
                loaded.append(name)
                logger.info("Plugin loaded: %s", name)
            else:
                logger.warning("Plugin %s: no register() function", name)
        except Exception as e:
            logger.warning("Plugin %s failed to load: %s", name, e)

    return loaded
