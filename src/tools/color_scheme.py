import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Union, Any


class ColorSchemeManager:
    SCIENTIFIC_PALETTES = {
        'sequential': {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'cividis': plt.cm.cividis,
            'Blues': plt.cm.Blues,
            'Greens': plt.cm.Greens,
            'Oranges': plt.cm.Oranges,
            'Reds': plt.cm.Reds,
            'YlOrBr': plt.cm.YlOrBr,
            'YlGn': plt.cm.YlGn,
            'BuGn': plt.cm.BuGn,
            'PuBu': plt.cm.PuBu,
            'GnBu': plt.cm.GnBu,
            'PuRd': plt.cm.PuRd,
            'RdPu': plt.cm.RdPu,
        },
        'diverging': {
            'RdBu_r': plt.cm.RdBu_r,
            'RdYlBu_r': plt.cm.RdYlBu_r,
            'seismic': plt.cm.seismic,
            'coolwarm': plt.cm.coolwarm,
            'bwr': plt.cm.bwr,
        },
        'qualitative': {
            'Set1': sns.color_palette('Set1'),
            'Set2': sns.color_palette('Set2'),
            'Set3': sns.color_palette('Set3'),
            'Paired': sns.color_palette('Paired'),
            'Dark2': sns.color_palette('Dark2'),
            'Accent': sns.color_palette('Accent'),
        }
    }

    EXPRESSION_PALETTES = {
        'heatmap_default': 'RdYlBu_r',
        'heatmap_blues': 'Blues',
        'heatmap_greenred': 'RdBu_r',
        'volcano_up': '#E41A1C',
        'volcano_down': '#377EB8',
        'volcano_ns': '#AAAAAA',
        'bar_expression': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        'scatter_default': '#4472C4',
    }

    @classmethod
    def get_palette(
        cls,
        palette_name: str,
        palette_type: str = 'sequential',
        n_colors: Optional[int] = None
    ) -> Union[mcolors.Colormap, List]:
        if palette_type in cls.SCIENTIFIC_PALETTES:
            palettes = cls.SCIENTIFIC_PALETTES[palette_type]
            if palette_name in palettes:
                cmap = palettes[palette_name]
                if n_colors and hasattr(cmap, '__call__'):
                    return cmap
                return cmap
        return plt.cm.viridis

    @classmethod
    def get_heatmap_colors(
        cls,
        data_min: float,
        data_max: float,
        palette: str = 'RdYlBu_r',
        center: Optional[float] = None
    ) -> mcolors.Normalize:
        if center is None:
            center = (data_min + data_max) / 2

        vmin = data_min
        vmax = data_max

        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    @classmethod
    def create_custom_palette(
        cls,
        colors: List[str],
        n_colors: Optional[int] = None
    ) -> List:
        if n_colors is None:
            return colors
        return sns.color_palette(colors, n_colors)

    @classmethod
    def get_volcano_colors(
        cls,
        log2fc: np.ndarray,
        pvalue: np.ndarray,
        fc_threshold: float = 1.0,
        pval_threshold: float = 0.05
    ) -> np.ndarray:
        colors = np.array([cls.EXPRESSION_PALETTES['volcano_ns']] * len(log2fc))

        sig_up = (log2fc >= fc_threshold) & (pvalue < pval_threshold)
        sig_down = (log2fc <= -fc_threshold) & (pvalue < pval_threshold)

        colors[sig_up] = cls.EXPRESSION_PALETTES['volcano_up']
        colors[sig_down] = cls.EXPRESSION_PALETTES['volcano_down']

        return colors

    @classmethod
    def list_available_palettes(cls) -> Dict[str, List[str]]:
        return {
            'sequential': list(cls.SCIENTIFIC_PALETTES['sequential'].keys()),
            'diverging': list(cls.SCIENTIFIC_PALETTES['diverging'].keys()),
            'qualitative': list(cls.SCIENTIFIC_PALETTES['qualitative'].keys()),
        }
