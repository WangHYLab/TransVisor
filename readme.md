# TransVisor - Transcriptome Visualization Agent

An intelligent transcriptome data visualization tool based on ReAct architecture, supporting natural language interaction for one-click generation of high-quality scientific charts.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

- **Natural Language Interaction**: Describe your visualization needs in English/Chinese, no coding required
- **Multiple Chart Types**: Bar plot, Scatter plot, Heatmap, Clustering heatmap, Volcano plot
- **Smart Data Processing**: Auto-detect gene/sample columns, supports multiple formats
- **Context Memory**: Multi-turn dialogue support, remembers loaded data
- **Scientific Color Schemes**: Professional color maps and palettes
- **High-Quality Output**: PNG, SVG formats, 300 DPI resolution

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Edit `config.yaml`:

```yaml
api:
  key: "your-deepseek-api-key"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
```

### Run Examples

```bash
# Basic usage
python -m src.cli --request "Create a bar plot for Gene_0001, Gene_0002"

# Interactive mode
python -m src.cli --interactive

# show help
python -m src.cli -h
```

## Usage

### Command Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--config` | string | Path to config file (default: config.yaml) | `--config my_config.yaml` |
| `--api-key` | string | DeepSeek API key (overrides config file) | `--api-key sk-xxxxx` |
| `--data` | string | Path to data file (CSV, TSV, Excel) | `--data data/expression.csv` |
| `--request` | string | Natural language visualization request | `--request "Create a bar plot"` |
| `--model` | string | Model to use (overrides config file) | `--model deepseek-chat` |
| `--max-iterations` | int | Maximum ReAct iterations | `--max-iterations 15` |
| `--show-full-steps` | flag | Show full ReAct execution steps | `--show-full-steps` |
| `--interactive` | flag | Run in interactive mode | `--interactive` |
| `--output-dir` | string | Output directory for visualizations | `--output-dir results` |
| `--verbose` | flag | Enable verbose output | `--verbose` |



## Supported Visualizations

| Chart Type | Description | Example Command |
|------------|-------------|----------------|
| **Bar Plot** | Gene expression comparison | `Create bar plot for Gene_0001, Gene_0002` |
| **Scatter Plot** | Gene correlation analysis | `Show correlation between Gene_0001 and Gene_0002` |
| **Heatmap** | Expression pattern visualization | `Create heatmap for top 20 genes` |
| **Clustering Heatmap** | Hierarchical clustering | `Create clustering heatmap` |
| **Volcano Plot** | Differential expression analysis | `Create volcano plot` |

## Examples

```bash
# Create gene expression heatmap
python -m src.cli --request "Create heatmap for Gene_0001, Gene_0003 to Gene_0010"

# Create volcano plot for differential expression
python -m src.cli --request "Create differential expression volcano plot" --data example_data/differential_expression.csv

# Create scatter plot for gene correlation
python -m src.cli --request "Create scatter plot for Gene_0001 and Gene_0002 correlation"

# Create bar plot for specific genes and samples
python -m src.cli --request "Create bar plot for Gene_0001 and Gene_0003 expression in first 5 samples"

# Create clustering heatmap
python -m src.cli --request "Create gene expression clustering heatmap"

# Create clustering heatmap for first 6 samples
python -m src.cli --request "Create clustering heatmap for first 6 samples"

# Use time series data
python -m src.cli --data example_data/time_series.csv --request "Create gene expression clustering heatmap" --show-full-steps
```

## Project Structure

```
Agent/
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── example_data/            # Sample data
│   ├── example_expression.csv
│   ├── differential_expression.csv
│   └── time_series.csv
├── output/                  # Output directory
├── src/                     # Source code
│   ├── cli.py               # CLI interface
│   ├── config.py            # Config parser
│   ├── main.py              # Agent entry
│   ├── prompts/             # Prompt engineering
│   ├── react_agent/         # ReAct core
│   ├── tools/               # Tool system
│   ├── utils/               # Utilities
│   └── visualizations/      # Visualization modules
└── tests/                   # Test files
```

## License

This project is licensed under the MIT License.