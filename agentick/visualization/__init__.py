"""Visualization and plotting for publication-quality figures."""


def __getattr__(name: str):
    if name == "AgentComparisonPlotter":
        from agentick.visualization.comparison_plots import AgentComparisonPlotter

        return AgentComparisonPlotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentComparisonPlotter"]
