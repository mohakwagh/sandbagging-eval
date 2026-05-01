import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

CONDITIONS = ["neutral", "subtle", "explicit"]
SANDBAGGING_CONDITIONS = ["subtle", "explicit"]
CONDITION_COLORS = {
    "neutral": "#4C9BE8",
    "subtle": "#F5A623",
    "explicit": "#E85454",
}


def generate_report(metrics: dict, output_path: str) -> None:
    """
    Generate a standalone HTML report with three visualizations:
    1. Grouped bar chart — accuracy by condition and category
    2. Bar chart — sandbagging rate (delta) by condition and category
    3. Summary table — overall sandbagging rate per condition
    """
    categories = sorted(metrics["accuracy"]["per_category"].keys())

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Accuracy by Condition and Category",
            "Sandbagging Rate by Condition and Category",
            "Overall Sandbagging Rate Summary",
        ),
        specs=[
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "table"}],
        ],
        vertical_spacing=0.1,
    )

    # Chart 1: accuracy per condition per category
    for condition in CONDITIONS:
        fig.add_trace(
            go.Bar(
                name=condition,
                x=categories,
                y=[
                    metrics["accuracy"]["per_category"][cat][condition]
                    for cat in categories
                ],
                marker_color=CONDITION_COLORS[condition],
                legendgroup=condition,
            ),
            row=1,
            col=1,
        )

    # Chart 2: sandbagging rate per condition per category
    for condition in SANDBAGGING_CONDITIONS:
        fig.add_trace(
            go.Bar(
                name=f"{condition} (delta)",
                x=categories,
                y=[
                    metrics["sandbagging_rate"]["per_category"][cat][condition]
                    for cat in categories
                ],
                marker_color=CONDITION_COLORS[condition],
                legendgroup=f"{condition}_delta",
            ),
            row=2,
            col=1,
        )

    # Chart 3: overall sandbagging rate summary table
    overall_sr = metrics["sandbagging_rate"]["overall"]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Condition", "Overall Sandbagging Rate"],
                fill_color="#2C3E50",
                font=dict(color="white", size=13),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[
                    SANDBAGGING_CONDITIONS,
                    [f"{overall_sr[c]:.4f}" for c in SANDBAGGING_CONDITIONS],
                ],
                fill_color=[["#F2F2F2", "#FFFFFF"] * 2],
                align="center",
                font=dict(size=12),
                height=28,
            ),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title_text="LLM Sandbagging Detection Report",
        barmode="group",
        height=1100,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Sandbagging Rate", row=2, col=1)

    pio.write_html(fig, output_path, full_html=True, include_plotlyjs=True)
