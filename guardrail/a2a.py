"""A lightweight Agent-to-Agent (A2A) messaging protocol.

Design mirrors real agent-communication standards (FIPA ACL performatives, and
the shape of Google's A2A / Anthropic MCP tool-messages): every interaction is a
typed, addressed, logged message rather than a hidden function call. That makes
the multi-agent deliberation auditable — which for a *security* control is the
whole point, and it drives the live-demo trace view.

    AgentMessage: one addressed speech-act between two agents.
    MessageBus:   routes messages, keeps the ordered conversation trace.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Performative(str, Enum):
    """The 'verb' of a message — what the sender intends by it."""
    REQUEST = "REQUEST"    # please do X
    INFORM = "INFORM"      # here is a result / fact
    ESCALATE = "ESCALATE"  # I'm not confident; you decide
    DECIDE = "DECIDE"      # final ruling
    LOG = "LOG"            # write to memory / threat intel


_ids = itertools.count(1)


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    performative: Performative
    content: dict[str, Any] = field(default_factory=dict)
    conversation_id: str = "conv"
    id: int = field(default_factory=lambda: next(_ids))
    ts: float = field(default_factory=time.perf_counter)

    def summary(self) -> str:
        """One-line human-readable form for the trace view."""
        keys = ", ".join(f"{k}={_short(v)}" for k, v in self.content.items())
        return f"{self.sender} --{self.performative.value}--> {self.receiver}  {{{keys}}}"


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 48 else s[:45] + "..."


class MessageBus:
    """Routes messages to registered agents and records the trace."""

    def __init__(self):
        self.agents: dict[str, Any] = {}
        self.trace: list[AgentMessage] = []

    def register(self, agent) -> None:
        self.agents[agent.name] = agent

    def send(self, msg: AgentMessage) -> list[AgentMessage]:
        """Deliver one message; return (and record) any replies the receiver produced."""
        self.trace.append(msg)
        receiver = self.agents.get(msg.receiver)
        if receiver is None:
            return []
        replies = receiver.handle(msg, self) or []
        self.trace.extend(replies)  # record the round-trip so the trace is complete
        return replies

    def reset(self) -> None:
        self.trace = []
