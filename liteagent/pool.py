"""Multi-agent pool with delegation support."""

import logging
from typing import Optional

from .agent import LiteAgent
from .config import get_soul_prompt

logger = logging.getLogger(__name__)


class AgentPool:
    """Manages named LiteAgent instances with cross-agent delegation."""

    def __init__(self, config: dict):
        self.config = config
        self._agents: dict[str, LiteAgent] = {}
        self._default_name: str = "main"

    @classmethod
    def from_config(cls, config: dict) -> "AgentPool":
        """Build pool from config. Falls back to single agent if no 'agents' key."""
        pool = cls(config)
        agents_cfg = config.get("agents", {})

        if not agents_cfg:
            agent = LiteAgent(config)
            pool._agents["main"] = agent
            return pool

        for name, agent_cfg in agents_cfg.items():
            # Merge agent-specific overrides with base config
            merged = {**config}
            if "soul" in agent_cfg:
                merged.setdefault("agent", {})
                merged["agent"] = {**merged.get("agent", {}), "soul": agent_cfg["soul"]}
            if "tools" in agent_cfg:
                merged["tools"] = agent_cfg["tools"]
            if "memory" in agent_cfg:
                merged["memory"] = {**config.get("memory", {}), **agent_cfg["memory"]}

            agent = LiteAgent(merged)
            pool._agents[name] = agent

            if agent_cfg.get("default", False) or name == "main":
                pool._default_name = name

        # Wire delegation tools
        pool._wire_delegation()
        logger.info("Agent pool created: %d agents (%s)",
                    len(pool._agents), ", ".join(pool._agents.keys()))
        return pool

    def _wire_delegation(self):
        """Add 'delegate' tool to each agent so they can call other agents."""
        if len(self._agents) < 2:
            return

        for name, agent in self._agents.items():
            other_names = [n for n in self._agents if n != name]
            descriptions = ", ".join(other_names)

            async def delegate_handler(agent_name: str, query: str,
                                       _pool=self) -> str:
                """Delegate a task to another agent.
                agent_name: Name of the agent to delegate to
                query: The task or question for the other agent"""
                target = _pool.get(agent_name)
                if not target:
                    return f"Unknown agent: {agent_name}. Available: {descriptions}"
                return await target.run(query)

            agent.tools._tools["delegate"] = {
                "name": "delegate",
                "description": f"Delegate a task to another agent. Available agents: {descriptions}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string",
                                       "description": f"One of: {descriptions}"},
                        "query": {"type": "string",
                                  "description": "Task for the other agent"},
                    },
                    "required": ["agent_name", "query"],
                },
            }
            agent.tools._handlers["delegate"] = delegate_handler

    def get(self, name: str | None = None) -> Optional[LiteAgent]:
        """Get agent by name. Returns default if name is None."""
        name = name or self._default_name
        return self._agents.get(name)

    @property
    def default(self) -> LiteAgent:
        """Get default agent."""
        return self._agents[self._default_name]

    def list_agents(self) -> list[str]:
        """Return list of agent names."""
        return list(self._agents.keys())

    def close_all(self):
        """Close all agent memory connections."""
        for agent in self._agents.values():
            try:
                agent.memory.close()
            except Exception:
                pass
