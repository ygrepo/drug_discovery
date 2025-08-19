from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
        # ax.set_xlim(-100, 100)
        # ax.set_ylim(-100, 100)
        # ax.set_xticks(np.arange(-100, 101, 20))
        # ax.set_yticks(np.arange(-100, 101, 20))
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
