from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from matplotlib.lines import Line2D

import logging

logger = logging.getLogger(__name__)


def plot_low_dim_panels(
    low_dim_sets: list[tuple[np.ndarray, np.ndarray]],
    *,
    titles: list[str] = None,
    figsize: tuple = (12, 10),
    xlims: list[tuple] = None,
    ylims: list[tuple] = None,
    xticks: list[tuple] = None,
    yticks: list[tuple] = None,
    xlabel: str = "t-SNE 1",
    ylabel: str = "t-SNE 2",
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    legend_fontsize: int = 9,
    save_path: Path = None,
    dpi: int = 300,
):
    """
    Plot four sets of low-dimensional wild-type vs mutant embeddings in a 2x2 layout.

    Parameters
    ----------
    low_dim_sets : list of tuple
        A list of 4 tuples, each (pt1_coords, pt2_coords), where:
          - pt1_coords: np.ndarray of shape (n, 2) for wild-type
          - pt2_coords: np.ndarray of shape (n, 2) for mutant
    titles : list of str, optional
        Titles for each subplot (length 4).
    figsize : tuple
        Figure size.
    """
    if titles is None:
        titles = [f"Panel {i+1}" for i in range(4)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (ax, (pt1_coords, pt2_coords)) in enumerate(zip(axes, low_dim_sets)):
        n = len(pt1_coords)

        # Orange connection lines
        for i in range(n):
            ax.plot(
                [pt1_coords[i, 0], pt2_coords[i, 0]],
                [pt1_coords[i, 1], pt2_coords[i, 1]],
                c="orange",
                lw=1,
            )

        # Wild-type points (circles)
        ax.scatter(
            pt1_coords[:, 0],
            pt1_coords[:, 1],
            c="#005f73",
            marker="o",
            label="Wild-type",
        )

        # Mutant points (crosses)
        ax.scatter(
            pt2_coords[:, 0], pt2_coords[:, 1], c="#fb7185", marker="x", label="Mutant"
        )

        if xlims is not None and xlims[idx] is not None:
            ax.set_xlim(xlims[idx])
        if ylims is not None and ylims[idx] is not None:
            ax.set_ylim(ylims[idx])
        if xticks is not None and xticks[idx] is not None:
            ax.set_xticks(xticks[idx])
        if yticks is not None and yticks[idx] is not None:
            ax.set_yticks(yticks[idx])

        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")
        ax.set_title(titles[idx], fontsize=title_fontsize, weight="bold")
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
        logger.info(f"Saved figure to {save_path}")
    plt.close(fig)


def plot_shared_embeddings(
    coords_2d: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    figsize: Tuple[int, int] = (9, 7),
    xlabel: str = "UMAP-1",
    ylabel: str = "UMAP-2",
    label_fontsize: int = 12,
    title_fontsize: int = 12,
    title: str = "WT vs Mut across Models (Shared UMAP)",
    # color per model (single hue); WT/Mut are distinguished by marker/face
    model_colors: Optional[Dict[str, str]] = None,
    # per-state styling
    wt_size: float = 18,
    mut_size: float = 28,
    wt_alpha: float = 0.9,
    mut_alpha: float = 0.9,
    wt_edge_lw: float = 0.7,
    mut_edge_lw: float = 1.0,
    # connection arrows
    draw_arrows: bool = True,
    arrow_alpha: float = 0.35,
    arrow_lw: float = 0.6,
    max_arrows_per_model: Optional[int] = 500,  # cap to avoid overplotting
    # whitespace control
    robust_limits: bool = True,
    limit_quantiles: Tuple[float, float] = (0.01, 0.99),
    margin: float = 0.05,  # 5% margins
    # legend/layout
    legend_fontsize: int = 12,
    save_path: Optional[Path] = None,
    dpi: int = 300,
    log_level: str = "INFO",
):
    """
    Plot all models in one shared UMAP space with strong WT/Mut distinction and compact axes.
    - Color encodes model, shape/face encodes state (WT vs Mut).
    - Axes limits use robust quantiles to reduce whitespace from outliers.
    """
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    model_names = list(coords_2d.keys())
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    # Default color cycle if not provided
    if model_colors is None:
        base = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        model_colors = {m: base[i % len(base)] for i, m in enumerate(model_names)}

    # Collect all coordinates for robust limits
    all_xy = []
    for m in model_names:
        wt2, mut2 = coords_2d[m]
        all_xy.append(wt2)
        all_xy.append(mut2)
    all_xy = np.vstack(all_xy) if len(all_xy) else np.zeros((0, 2))

    # Draw arrows first (behind points)
    if draw_arrows:
        for m in model_names:
            wt2, mut2 = coords_2d[m]
            n_pairs = min(len(wt2), len(mut2))
            if max_arrows_per_model is not None:
                n_pairs = min(n_pairs, max_arrows_per_model)
            color = model_colors[m]
            # sample evenly if too many
            if len(wt2) > n_pairs:
                idx = np.linspace(0, len(wt2) - 1, n_pairs, dtype=int)
            else:
                idx = np.arange(n_pairs)
            for i in idx:
                ax.arrow(
                    wt2[i, 0],
                    wt2[i, 1],
                    mut2[i, 0] - wt2[i, 0],
                    mut2[i, 1] - wt2[i, 1],
                    head_width=0.0,
                    length_includes_head=True,
                    lw=arrow_lw,
                    alpha=arrow_alpha,
                    color=color,
                    zorder=1,
                )

    # Plot points: WT as filled circles with white edge; Mut as hollow squares with colored edge
    handles_models = []
    for m in model_names:
        wt2, mut2 = coords_2d[m]
        color = model_colors[m]

        ax.scatter(
            wt2[:, 0],
            wt2[:, 1],
            s=wt_size,
            marker="o",
            facecolors=color,
            edgecolors="white",
            linewidths=wt_edge_lw,
            alpha=wt_alpha,
            zorder=3,
            label=f"{m} WT",
        )
        ax.scatter(
            mut2[:, 0],
            mut2[:, 1],
            s=mut_size,
            marker="s",
            facecolors="none",
            edgecolors=color,
            linewidths=mut_edge_lw,
            alpha=mut_alpha,
            zorder=4,
            label=f"{m} Mut",
        )
        # Keep one handle per model (use WT handle for color swatch)
        handles_models.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                label=m,
            )
        )

    # Axes cosmetics
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=label_fontsize)
    ax.set_title(title, fontweight="bold", pad=10, fontsize=title_fontsize)

    # Robust limits to reduce whitespace
    if robust_limits and all_xy.size:
        qx = np.quantile(all_xy[:, 0], limit_quantiles)
        qy = np.quantile(all_xy[:, 1], limit_quantiles)
        xr = qx[1] - qx[0]
        yr = qy[1] - qy[0]
        ax.set_xlim(qx[0] - xr * margin, qx[1] + xr * margin)
        ax.set_ylim(qy[0] - yr * margin, qy[1] + yr * margin)
    else:
        ax.margins(margin)

    # Cleaner frame
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)

    # Two legends: (1) models by color, (2) state by marker
    # Legend 1: models
    leg1 = ax.legend(
        handles=handles_models,
        title="Model",
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    ax.add_artist(leg1)

    # Legend 2: state
    state_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor="#666666",
            markeredgecolor="white",
            label="WT",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="#666666",
            label="Mut",
        ),
    ]
    ax.legend(
        handles=state_handles,
        title="WT vs. Mutant",
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.63),
    )

    plt.tight_layout(rect=(0, 0.16, 1, 1))

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    plt.show()
    plt.close(fig)
