import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from ..tools.plotter import Plotter
from ..tools.color_scheme import ColorSchemeManager


def create_bar_plot(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = "Gene Expression Bar Plot",
    xlabel: str = "Genes",
    ylabel: str = "Expression Level",
    palette: str = "Set2",
    figsize: Tuple[int, int] = (5, 5),
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    plotter = Plotter()
    result = plotter.bar_plot(
        data=data,
        x_column=x_column,
        y_columns=y_columns,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        palette=palette,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )
    return {
        "success": True,
        "plot_result": result,
        "message": f"Bar plot created: {title}"
    }


def create_scatter_plot(
    x_data: Union[List[float], np.ndarray],
    y_data: Union[List[float], np.ndarray],
    title: str = "Gene Expression Correlation",
    xlabel: str = "Gene A Expression",
    ylabel: str = "Gene B Expression",
    color: str = "#4472C4",
    figsize: Tuple[int, int] = (5, 5),
    show_r2: bool = True,
    regression_line: bool = True,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    plotter = Plotter()
    result = plotter.scatter_plot(
        x_data=x_data,
        y_data=y_data,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        color=color,
        figsize=figsize,
        show_r2=show_r2,
        regression_line=regression_line,
        save_path=save_path,
        **kwargs
    )

    correlation = np.corrcoef(x_data, y_data)[0, 1]

    return {
        "success": True,
        "plot_result": result,
        "correlation": float(correlation),
        "r_squared": float(correlation ** 2),
        "message": f"Scatter plot created: {title}"
    }


def create_heatmap(
    data: Union[np.ndarray, pd.DataFrame],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Gene Expression Heatmap",
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (6, 6),
    show_values: bool = False,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    if isinstance(data, pd.DataFrame):
        data_matrix = data.values
        if row_labels is None:
            row_labels = data.index.tolist() if hasattr(data, 'index') else None
        if col_labels is None:
            col_labels = data.columns.tolist()
    else:
        data_matrix = data

    plotter = Plotter()
    result = plotter.heatmap(
        data=data_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=title,
        cmap=cmap,
        figsize=figsize,
        show_values=show_values,
        save_path=save_path,
        **kwargs
    )

    return {
        "success": True,
        "plot_result": result,
        "message": f"Heatmap created: {title}"
    }


def create_clustering_heatmap(
    data: Union[np.ndarray, pd.DataFrame],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: str = "Hierarchical Clustering Heatmap",
    metric: str = 'euclidean',
    method: str = 'ward',
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    if isinstance(data, pd.DataFrame):
        data_matrix = data.values
        if row_labels is None:
            row_labels = data.index.tolist() if hasattr(data, 'index') else None
        if col_labels is None:
            col_labels = data.columns.tolist()
    else:
        data_matrix = data

    plotter = Plotter()
    result = plotter.clustering_heatmap(
        data=data_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=title,
        metric=metric,
        method=method,
        cmap=cmap,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )

    return {
        "success": True,
        "plot_result": result,
        "message": f"Clustering heatmap created: {title}"
    }


def create_volcano_plot(
    log2fc: Union[List[float], np.ndarray],
    pvalue: Union[List[float], np.ndarray],
    gene_labels: Optional[List[str]] = None,
    fc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    title: str = "Volcano Plot",
    xlabel: str = "log2 Fold Change",
    ylabel: str = "-log10(p-value)",
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (5, 5),
    label_top_genes: int = 10,
    save_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    log2fc = np.array(log2fc)
    pvalue = np.array(pvalue)

    plotter = Plotter()
    result = plotter.volcano_plot(
        log2fc=log2fc,
        pvalue=pvalue,
        gene_labels=gene_labels,
        fc_threshold=fc_threshold,
        pval_threshold=pval_threshold,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        colors=colors,
        figsize=figsize,
        label_top_genes=label_top_genes,
        save_path=save_path,
        **kwargs
    )

    return {
        "success": True,
        "plot_result": result,
        "upregulated_genes": result['sig_up_count'],
        "downregulated_genes": result['sig_down_count'],
        "message": f"Volcano plot created: {title}"
    }
