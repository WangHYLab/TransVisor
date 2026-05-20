import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


class Plotter:
    def __init__(self, style: str = 'seaborn'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figure = None
        self.axes = None

    def bar_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        title: str = "Gene Expression Bar Plot",
        xlabel: str = "Genes",
        ylabel: str = "Expression Level",
        palette: str = "Set2",
        figsize: Tuple[int, int] = (12, 6),
        rotation: int = 45,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        fig, ax = plt.subplots(figsize=figsize)

        plot_data = data[[x_column] + y_columns].melt(
            id_vars=[x_column],
            var_name='Condition',
            value_name='Expression'
        )

        sns.barplot(
            data=plot_data,
            x=x_column,
            y='Expression',
            hue='Condition',
            palette=palette,
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='x', rotation=rotation)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        result = {'figure': fig, 'axes': ax, 'data': plot_data}
        self.figure = fig
        self.axes = ax
        return result

    def scatter_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Gene Expression Correlation",
        xlabel: str = "Gene A Expression",
        ylabel: str = "Gene B Expression",
        color: str = "#4472C4",
        figsize: Tuple[int, int] = (10, 8),
        show_r2: bool = True,
        regression_line: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(x_data, y_data, c=color, alpha=0.6, s=50, edgecolors='white')

        if regression_line:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=2)

        if show_r2:
            correlation = np.corrcoef(x_data, y_data)[0, 1]
            r2 = correlation ** 2
            ax.text(
                0.05, 0.95,
                f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top'
            )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        result = {'figure': fig, 'axes': ax}
        self.figure = fig
        self.axes = ax
        return result

    def heatmap(
        self,
        data: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = "Gene Expression Heatmap",
        cmap: str = "RdYlBu_r",
        figsize: Tuple[int, int] = (12, 10),
        show_values: bool = False,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            data,
            cmap=cmap,
            xticklabels=col_labels,
            yticklabels=row_labels,
            ax=ax,
            **kwargs
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        result = {'figure': fig, 'axes': ax}
        self.figure = fig
        self.axes = ax
        return result

    def clustering_heatmap(
        self,
        data: np.ndarray,
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: str = "Hierarchical Clustering Heatmap",
        metric: str = 'euclidean',
        method: str = 'ward',
        cmap: str = "RdYlBu_r",
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        g = sns.clustermap(
            data,
            cmap=cmap,
            xticklabels=col_labels,
            yticklabels=row_labels,
            method=method,
            metric=metric,
            figsize=figsize,
            dendrogram_ratio=0.15,
            cbar_pos=(0.02, 0.8, 0.03, 0.15)
        )

        g.ax_heatmap.set_title(title, fontsize=14, fontweight='bold', pad=10)

        if save_path:
            g.savefig(save_path, dpi=300, bbox_inches='tight')

        result = {'figure': g.figure, 'axes': g.ax_heatmap, 'clustermap': g}
        self.figure = g.figure
        self.axes = g.ax_heatmap
        return result

    def volcano_plot(
        self,
        log2fc: np.ndarray,
        pvalue: np.ndarray,
        gene_labels: Optional[List[str]] = None,
        fc_threshold: float = 1.0,
        pval_threshold: float = 0.05,
        title: str = "Volcano Plot",
        xlabel: str = "log2 Fold Change",
        ylabel: str = "-log10(p-value)",
        colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        label_top_genes: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        if colors is None:
            colors = {
                'up': '#E41A1C',
                'down': '#377EB8',
                'ns': '#AAAAAA'
            }

        neg_log_pval = -np.log10(pvalue + 1e-300)

        fig, ax = plt.subplots(figsize=figsize)

        sig_up = (log2fc >= fc_threshold) & (pvalue < pval_threshold)
        sig_down = (log2fc <= -fc_threshold) & (pvalue < pval_threshold)
        not_sig = ~(sig_up | sig_down)

        ax.scatter(log2fc[not_sig], neg_log_pval[not_sig],
                   c=colors['ns'], alpha=0.5, s=30, label='Not Significant')
        ax.scatter(log2fc[sig_up], neg_log_pval[sig_up],
                   c=colors['up'], alpha=0.7, s=40, label='Up-regulated')
        ax.scatter(log2fc[sig_down], neg_log_pval[sig_down],
                   c=colors['down'], alpha=0.7, s=40, label='Down-regulated')

        ax.axhline(y=-np.log10(pval_threshold), color='gray',
                   linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=fc_threshold, color='gray',
                   linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=-fc_threshold, color='gray',
                   linestyle='--', linewidth=1, alpha=0.7)

        if label_top_genes > 0 and gene_labels is not None:
            top_genes_idx = np.argsort(pvalue)[:label_top_genes]
            for idx in top_genes_idx:
                ax.annotate(
                    gene_labels[idx],
                    (log2fc[idx], neg_log_pval[idx]),
                    fontsize=8,
                    alpha=0.8
                )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        result = {
            'figure': fig,
            'axes': ax,
            'sig_up_count': np.sum(sig_up),
            'sig_down_count': np.sum(sig_down),
            'not_sig_count': np.sum(not_sig)
        }
        self.figure = fig
        self.axes = ax
        return result

    @staticmethod
    def close_all():
        plt.close('all')
