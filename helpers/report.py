from typing import List

from plotly import graph_objects as go


def save_plots(figures: List[go.Figure], filename: str) -> None:
    with open(filename, "w") as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
