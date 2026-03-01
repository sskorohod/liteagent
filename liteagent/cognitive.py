"""Cognitive self-refinement tools: planning and active memory management."""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def register_cognitive_tools(agent):
    """Register propose_plan and reconcile_knowledge tools."""
    
    # 1. propose_plan
    async def propose_plan(steps: list[str]) -> str:
        """Propose a multi-step plan for a complex task.
        
        steps: List of logical steps to accomplish the goal.
        """
        plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        # Store in session state via memory system
        await agent.memory.set_state("session:current_plan", plan_text, agent._current_user_id)
        logger.info("Plan proposed: %d steps", len(steps))
        return f"Plan accepted and stored:\n{plan_text}\n\nProceed with the first step."

    agent.tools.register_tool(
        name="propose_plan",
        description="Propose a multi-step plan for complex tasks. Use this before taking actions.",
        input_schema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of steps to take"
                }
            },
            "required": ["steps"]
        },
        handler=propose_plan
    )

    # 2. reconcile_knowledge
    async def reconcile_knowledge(fact: str, replaces: str = None) -> str:
        """Update or correct a fact in long-term memory.
        
        fact: The new/corrected information.
        replaces: (Optional) Fragment of the old fact to be marked as obsolete.
        """
        # If replaces is provided, we simulate a "forget" followed by a "remember"
        if replaces:
            agent.memory.forget(agent._current_user_id, replaces)
        
        # Store new fact with high importance since it was explicitly reconciled
        await agent.memory.remember(fact, agent._current_user_id, memory_type="fact", importance=0.9)
        
        logger.info("Knowledge reconciled: %s", fact[:50])
        return f"Memory updated: '{fact}'. " + (f"Obsolete information matching '{replaces}' has been removed." if replaces else "")

    agent.tools.register_tool(
        name="reconcile_knowledge",
        description="Update or correct a fact in long-term memory. Use this when you discover new information that contradicts old knowledge.",
        input_schema={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The new or corrected fact"},
                "replaces": {"type": "string", "description": "Optional fragment of the OLD fact that should be forgotten"}
            },
            "required": ["fact"]
        },
        handler=reconcile_knowledge
    )
