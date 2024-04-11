from typing import List

import dominate
from plotly import graph_objects as go


def generate_report(
    figures: List[go.Figure], headers: List[str], filename: str
) -> None:
    if len(figures) != len(headers):
        raise ValueError("Number of figures and headers must be the same")

    doc = dominate.document(title="Report")
    with doc:
        for figure_name, figure in zip(headers, figures):
            dominate.tags.h1(figure_name)
            dominate.tags.div(figure.to_html(full_html=False, include_plotlyjs="cdn"))
    with open(filename, "w") as f:
        f.write(doc.render())
