from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """Metadata describing the kind of operation a Tool performs."""

    WORKSPACE_READ = "workspace_read"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS = "process"
    SCRIPT_EXECUTION = "script_execution"
    CAPABILITY_MANAGEMENT = "capability_management"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    WEB_SEARCH = "web_search"
    AGENT_CONTROL = "agent_control"
