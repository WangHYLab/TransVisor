import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.file_reader import FileReader
from src.tools.preprocessor import DataPreprocessor
from src.tools.color_scheme import ColorSchemeManager
from src.tools.plotter import Plotter


class TestFileReader:
    def test_read_csv(self, tmp_path):
        df = pd.DataFrame({
            'gene': ['Gene1', 'Gene2', 'Gene3'],
            'sample1': [10.5, 20.3, 15.7],
            'sample2': [12.1, 22.4, 18.9]
        })
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)

        result = FileReader.read_csv(str(file_path))
        assert result.shape == (3, 3)
        assert list(result.columns) == ['gene', 'sample1', 'sample2']

    def test_read_tsv(self, tmp_path):
        df = pd.DataFrame({
            'gene': ['Gene1', 'Gene2'],
            'exp1': [1.0, 2.0],
            'exp2': [3.0, 4.0]
        })
        file_path = tmp_path / "test.tsv"
        df.to_csv(file_path, index=False, sep='\t')

        result = FileReader.read_tsv(str(file_path))
        assert result.shape == (2, 3)

    def test_get_data_info(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'exp1': [1.0, 2.0, np.nan],
            'exp2': [4.0, 5.0, 6.0]
        })

        info = FileReader.get_data_info(df)

        assert info['shape'] == (3, 3)
        assert len(info['columns']) == 3
        assert info['missing_values']['exp1'] == 1

    def test_validate_expression_data(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'exp1': [1.0, 2.0, 3.0],
            'exp2': [4.0, 5.0, 6.0]
        })

        validation = FileReader.validate_expression_data(df)

        assert validation['valid'] == True
        assert validation['sample_column'] == 'exp1'
        assert 'expression_range' in validation


class TestDataPreprocessor:
    def test_clean_data_remove_na(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'exp': [1.0, np.nan, 3.0]
        })

        result = DataPreprocessor.clean_data(df, remove_na=True)
        assert result.shape[0] == 2

    def test_normalize_log2(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2'],
            'exp1': [4.0, 8.0],
            'exp2': [2.0, 4.0]
        })

        result = DataPreprocessor.normalize(df, method='log2')
        assert result['exp1'].iloc[0] == np.log2(5.0)
        assert result['exp2'].iloc[0] == np.log2(3.0)

    def test_filter_genes_min_expression(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'exp1': [1.0, 10.0, 5.0],
            'exp2': [2.0, 15.0, 3.0]
        })

        result = DataPreprocessor.filter_genes(
            df,
            gene_column='gene',
            min_expression=8.0
        )
        assert len(result) <= 2

    def test_impute_missing_values(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'exp': [1.0, np.nan, 3.0]
        })

        result = DataPreprocessor.impute_missing_values(df, method='mean')
        assert not result['exp'].isnull().any()


class TestColorSchemeManager:
    def test_get_palette(self):
        cmap = ColorSchemeManager.get_palette('viridis', 'sequential')
        assert cmap is not None

    def test_list_available_palettes(self):
        palettes = ColorSchemeManager.list_available_palettes()
        assert 'sequential' in palettes
        assert 'diverging' in palettes
        assert 'qualitative' in palettes

    def test_get_heatmap_colors(self):
        norm = ColorSchemeManager.get_heatmap_colors(0, 10, 'RdYlBu_r')
        assert norm(5) == 0.5


class TestPlotter:
    def test_bar_plot_creation(self):
        df = pd.DataFrame({
            'gene': ['G1', 'G2', 'G3'],
            'control': [1.0, 2.0, 3.0],
            'treatment': [1.5, 2.5, 3.5]
        })

        plotter = Plotter()
        result = plotter.bar_plot(
            data=df,
            x_column='gene',
            y_columns=['control', 'treatment'],
            title='Test Bar Plot'
        )

        assert 'figure' in result
        assert 'axes' in result
        plotter.close_all()

    def test_scatter_plot_creation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        plotter = Plotter()
        result = plotter.scatter_plot(x, y, title='Test Scatter')

        assert 'figure' in result
        assert 'axes' in result
        plotter.close_all()

    def test_heatmap_creation(self):
        data = np.random.rand(5, 4)

        plotter = Plotter()
        result = plotter.heatmap(
            data,
            row_labels=['G1', 'G2', 'G3', 'G4', 'G5'],
            col_labels=['S1', 'S2', 'S3', 'S4'],
            title='Test Heatmap'
        )

        assert 'figure' in result
        plotter.close_all()

    def test_volcano_plot_creation(self):
        log2fc = np.random.normal(0, 1.5, 100)
        pvalue = np.random.uniform(0, 0.1, 100)

        plotter = Plotter()
        result = plotter.volcano_plot(
            log2fc, pvalue,
            fc_threshold=1.0,
            pval_threshold=0.05
        )

        assert 'figure' in result
        assert 'sig_up_count' in result
        assert 'sig_down_count' in result
        plotter.close_all()


class TestReActArchitecture:
    def test_thought_creation(self):
        from src.react_agent.core import Thought

        thought = Thought(content="Test thought")
        assert thought.content == "Test thought"
        assert thought.timestamp is not None

    def test_action_creation(self):
        from src.react_agent.core import Action

        action = Action(
            tool_name="test_tool",
            tool_input={"param": "value"}
        )
        assert action.tool_name == "test_tool"
        assert action.tool_input["param"] == "value"

    def test_observation_creation(self):
        from src.react_agent.core import Observation

        obs = Observation(
            tool_name="test_tool",
            result={"output": "result"},
            success=True
        )
        assert obs.success == True
        assert obs.result["output"] == "result"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
