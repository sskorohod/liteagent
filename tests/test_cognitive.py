import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from liteagent.agent import LiteAgent
from liteagent.cognitive import register_cognitive_tools

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.memory = MagicMock()
    agent.memory.db = MagicMock()
    agent.memory.set_state = AsyncMock()
    agent.memory.remember = AsyncMock()
    agent.memory.forget = MagicMock()
    agent.tools = MagicMock()
    agent.tools._tools = {}
    agent.tools._handlers = {}
    
    # Mock register_tool to emulate ToolRegistry behavior
    def register_tool(name, description, input_schema, handler):
        agent.tools._tools[name] = {"name": name, "description": description, "input_schema": input_schema}
        agent.tools._handlers[name] = handler
    
    agent.tools.register_tool = register_tool
    agent._current_user_id = "test_user"
    return agent

@pytest.mark.asyncio
async def test_propose_plan(mock_agent):
    register_cognitive_tools(mock_agent)
    
    handler = mock_agent.tools._handlers["propose_plan"]
    steps = ["Step 1", "Step 2"]
    
    result = await handler(steps=steps)
    
    assert "Plan accepted" in result
    mock_agent.memory.set_state.assert_called_once()
    args, _ = mock_agent.memory.set_state.call_args
    assert "session:current_plan" in args
    assert "1. Step 1" in args[1]

@pytest.mark.asyncio
async def test_reconcile_knowledge(mock_agent):
    register_cognitive_tools(mock_agent)
    
    handler = mock_agent.tools._handlers["reconcile_knowledge"]
    
    # Test simple remember
    await handler(fact="User lives in Paris")
    mock_agent.memory.remember.assert_called_once_with(
        "User lives in Paris", "test_user", memory_type="fact", importance=0.9
    )
    
    # Test reconciliation (forget + remember)
    mock_agent.memory.remember.reset_mock()
    await handler(fact="User lives in London", replaces="Paris")
    
    mock_agent.memory.forget.assert_called_once_with("test_user", "Paris")
    mock_agent.memory.remember.assert_called_once_with(
        "User lives in London", "test_user", memory_type="fact", importance=0.9
    )
