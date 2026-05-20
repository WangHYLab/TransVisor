import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union


class ImageOptimizer:
    @staticmethod
    def add_title(
        axes,
        title: str,
        fontsize: int = 14,
        fontweight: str = 'bold',
        pad: float = 20,
        **kwargs
    ):
        axes.set_title(
            title,
            fontsize=fontsize,
            fontweight=fontweight,
            pad=pad,
            **kwargs
        )

    @staticmethod
    def add_labels(
        axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        if xlabel:
            axes.set_xlabel(xlabel, fontsize=12, **kwargs)
        if ylabel:
            axes.set_ylabel(ylabel, fontsize=12, **kwargs)
        if title:
            axes.set_title(title, fontsize=14, fontweight='bold', **kwargs)

    @staticmethod
    def adjust_layout(
        figure: Figure,
        left: float = 0.1,
        right: float = 0.9,
        top: float = 0.9,
        bottom: float = 0.1,
        wspace: float = 0.2,
        hspace: float = 0.2
    ):
        figure.subplots_adjust(
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            wspace=wspace,
            hspace=hspace
        )

    @staticmethod
    def set_axis_scale(
        axes,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None
    ):
        if xscale:
            axes.set_xscale(xscale)
        if yscale:
            axes.set_yscale(yscale)
        if xlim:
            axes.set_xlim(xlim)
        if ylim:
            axes.set_ylim(ylim)

    @staticmethod
    def add_legend(
        axes,
        location: str = 'upper right',
        frameon: bool = True,
        fontsize: int = 10,
        **kwargs
    ):
        axes.legend(
            loc=location,
            frameon=frameon,
            fontsize=fontsize,
            **kwargs
        )

    @staticmethod
    def add_grid(
        axes,
        which: str = 'major',
        axis: str = 'both',
        linestyle: str = '--',
        alpha: float = 0.5,
        **kwargs
    ):
        axes.grid(
            which=which,
            axis=axis,
            linestyle=linestyle,
            alpha=alpha,
            **kwargs
        )

    @staticmethod
    def rotate_labels(
        axes,
        x_rotation: int = 45,
        y_rotation: int = 0,
        **kwargs
    ):
        for label in axes.get_xticklabels():
            label.set_rotation(x_rotation)
        for label in axes.get_yticklabels():
            label.set_rotation(y_rotation)

    @staticmethod
    def set_figure_size(
        figure: Figure,
        width: float,
        height: float
):
        figure.set_size_inches(width, height)

    @staticmethod
    def apply_style(
        axes,
        spine_visible: bool = True,
        tick_params_visible: bool = True,
        grid_visible: bool = False
    ):
        for spine in axes.spines.values():
            spine.set_visible(spine_visible)

        if not tick_params_visible:
            axes.tick_params(
                left=False,
                bottom=False,
                top=False,
                right=False,
                labelleft=False,
                labelbottom=False
            )

        if grid_visible:
            axes.grid(True, alpha=0.3)
