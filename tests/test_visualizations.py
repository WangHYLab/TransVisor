import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizations.plots import (
    create_bar_plot,
    create_scatter_plot,
    create_heatmap,
    create_clustering_heatmap,
    create_volcano_plot
)


class TestVisualizationFunctions:
    def test_create_bar_plot(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'ctrl': [1.0, 2.0, 3.0],
            'treat': [1.5, 2.5, 3.5]
        })

        result = create_bar_plot(
            data=df,
            x_column='gene',
            y_columns=['ctrl', 'treat'],
            title='Test'
        )

        assert result['success'] == True
        assert 'plot_result' in result

    def test_create_scatter_plot(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 4.0, 5.9, 8.1, 10.2])

        result = create_scatter_plot(
            x_data=x,
            y_data=y,
            title='Correlation Test'
        )

        assert result['success'] == True
        assert 'correlation' in result
        assert 'r_squared' in result
        assert result['correlation'] > 0.9

    def test_create_heatmap(self):
        data = np.random.rand(10, 8)

        result = create_heatmap(
            data=data,
            row_labels=[f'G{i}' for i in range(10)],
            col_labels=[f'S{i}' for i in range(8)],
            title='Expression Heatmap'
        )

        assert result['success'] == True

    def test_create_volcano_plot(self):
        log2fc = np.concatenate([
            np.random.normal(3, 0.5, 50),
            np.random.normal(-3, 0.5, 50),
            np.random.normal(0, 0.5, 900)
        ])
        pvalue = np.concatenate([
            np.random.uniform(0.0001, 0.01, 50),
            np.random.uniform(0.0001, 0.01, 50),
            np.random.uniform(0.1, 1, 900)
        ])

        result = create_volcano_plot(
            log2fc=log2fc,
            pvalue=pvalue,
            gene_labels=[f'G{i}' for i in range(1000)],
            fc_threshold=1.0,
            pval_threshold=0.05
        )

        assert result['success'] == True
        assert result['upregulated_genes'] >= 40
        assert result['downregulated_genes'] >= 40

    def test_create_clustering_heatmap(self):
        np.random.seed(42)
        data = np.random.rand(8, 6)

        result = create_clustering_heatmap(
            data=data,
            row_labels=[f'Gene{i}' for i in range(8)],
            col_labels=[f'Sample{i}' for i in range(6)],
            title='Clustered Heatmap'
        )

        assert result['success'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
