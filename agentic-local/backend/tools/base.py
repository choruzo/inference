from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_for_prompt(self) -> str:
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  schema: {tool.parameters}")
        return "\n".join(lines)

    def run(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.get(name)
        if not tool:
            known = ", ".join(sorted(self._tools))
            raise ValueError(f"Unknown tool '{name}'. Available tools: {known}")
        return tool.handler(args)
