"""
Agents package for insurance policy debate system.
"""
from .debate_agents import (
    DebateAgent,
    AgentRole,
    create_debate_agents,
    create_debate_agents_with_judge,
    create_insurance_debate_agents  # Legacy compatibility
)
from .orchestrator import (
    DebateOrchestrator,
    quick_debate
)

__all__ = [
    'DebateAgent',
    'AgentRole',
    'create_debate_agents',
    'create_debate_agents_with_judge',
    'create_insurance_debate_agents',
    'DebateOrchestrator',
    'quick_debate'
]
