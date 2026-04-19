"""
Example ulcagent plugin.

Drop this file in the plugins/ directory. It auto-registers a custom tool.
Rename and modify for your own tools.

Each plugin must export a register(registry) function that adds ToolSchema entries.
"""


def register(registry):
    """Called by ulcagent on startup. Add your custom tools here."""
    from engine.agent_tools import ToolSchema

    def _hello(name: str = "world") -> str:
        return f"Hello, {name}! This is a custom plugin tool."

    registry.register(ToolSchema(
        name="hello",
        description="Example plugin tool — says hello. Replace with your own logic.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Who to greet", "default": "world"},
            },
        },
        function=lambda name="world": _hello(name),
        category="plugin",
    ))
