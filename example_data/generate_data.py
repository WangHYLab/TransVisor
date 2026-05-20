import pandas as pd
import numpy as np
from pathlib import Path
import os


def generate_example_expression_data(
    n_genes: int = 100,
    n_samples: int = 12,
    output_path: str = "example_data/example_expression.csv"
):
    np.random.seed(42)

    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    groups = ['Control'] * 6 + ['Treatment'] * 6
    group_labels = [f"{sample_names[i]}_{groups[i]}" for i in range(n_samples)]

    data = {
        'gene': gene_names
    }

    for i, sample in enumerate(sample_names):
        if groups[i] == 'Control':
            base_expression = np.random.lognormal(mean=2, sigma=0.5, size=n_genes)
        else:
            base_expression = np.random.lognormal(mean=3, sigma=0.7, size=n_genes)
            upregulated_genes = np.random.choice(n_genes, size=int(n_genes * 0.1), replace=False)
            base_expression[upregulated_genes] *= 3
            downregulated_genes = np.random.choice(
                [j for j in range(n_genes) if j not in upregulated_genes],
                size=int(n_genes * 0.05),
                replace=False
            )
            base_expression[downregulated_genes] *= 0.3

        data[sample] = base_expression

    df = pd.DataFrame(data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Generated expression data: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Genes: {n_genes}")
    print(f"  Samples: {n_samples} ({groups.count('Control')} Control, {groups.count('Treatment')} Treatment)")

    return df


def generate_differential_expression_data(
    n_genes: int = 500,
    output_path: str = "example_data/differential_expression.csv"
):
    np.random.seed(123)

    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]

    log2fc = np.random.normal(0, 1.5, n_genes)
    pvalue = np.random.uniform(0, 1, n_genes)

    significant_up = np.random.choice(n_genes, size=int(n_genes * 0.08), replace=False)
    log2fc[significant_up] = np.abs(np.random.normal(2.5, 0.5, len(significant_up)))
    pvalue[significant_up] = np.random.uniform(0.0001, 0.01, len(significant_up))

    significant_down = np.random.choice(
        [i for i in range(n_genes) if i not in significant_up],
        size=int(n_genes * 0.06),
        replace=False
    )
    log2fc[significant_down] = -np.abs(np.random.normal(-2.5, 0.5, len(significant_down)))
    pvalue[significant_down] = np.random.uniform(0.0001, 0.01, len(significant_down))

    df = pd.DataFrame({
        'gene': gene_names,
        'log2FoldChange': log2fc,
        'pvalue': pvalue,
        'padj': np.minimum(pvalue * 10, 1.0)
    })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Generated differential expression data: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Up-regulated: {len(significant_up)}")
    print(f"  Down-regulated: {len(significant_down)}")

    return df


def generate_time_series_data(
    n_genes: int = 30,
    time_points: int = 6,
    output_path: str = "example_data/time_series.csv"
):
    np.random.seed(456)

    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
    time_labels = [f"T{i}h" for i in range(time_points)]

    data = {'gene': gene_names}

    for t in range(time_points):
        time_factor = t / (time_points - 1)
        expressions = []

        for g in range(n_genes):
            if g < n_genes // 3:
                base = 2 + 3 * time_factor + np.random.normal(0, 0.3)
            elif g < 2 * n_genes // 3:
                base = 3 + 2 * np.sin(time_factor * np.pi) + np.random.normal(0, 0.3)
            else:
                base = 4 - 2 * time_factor + np.random.normal(0, 0.3)
            expressions.append(max(0.1, base))

        data[f'T{t}h'] = expressions

    df = pd.DataFrame(data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Generated time series data: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Time points: {time_points}")

    return df


if __name__ == '__main__':
    print("Generating example data files...")
    print("-" * 40)

    generate_example_expression_data()
    generate_differential_expression_data()
    generate_time_series_data()

    print("-" * 40)
    print("All example data files generated successfully!")
