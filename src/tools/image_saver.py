import matplotlib.pyplot as plt
from typing import Optional, Union, List
import os


class ImageSaver:
    SUPPORTED_FORMATS = ['png', 'svg', 'pdf', 'jpg', 'eps']

    @staticmethod
    def save_figure(
        figure: plt.Figure,
        file_path: str,
        formats: Optional[Union[str, List[str]]] = None,
        dpi: int = 300,
        transparent: bool = False,
        **kwargs
    ) -> List[str]:
        if formats is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext:
                formats = [ext.strip('.')]
            else:
                formats = ['png']
        elif isinstance(formats, str):
            formats = [formats]

        saved_paths = []
        base_path = os.path.splitext(file_path)[0]

        for fmt in formats:
            if fmt not in ImageSaver.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format: {fmt}. "
                    f"Supported: {ImageSaver.SUPPORTED_FORMATS}"
                )

            save_path = f"{base_path}.{fmt}"
            figure.savefig(
                save_path,
                format=fmt,
                dpi=dpi,
                transparent=transparent,
                bbox_inches='tight',
                **kwargs
            )
            saved_paths.append(save_path)

        return saved_paths

    @staticmethod
    def save_axes(
        axes,
        file_path: str,
        format: str = 'png',
        dpi: int = 300,
        **kwargs
    ) -> str:
        figure = axes.get_figure()
        if figure is None:
            raise ValueError("Could not get figure from axes")

        figure.savefig(
            file_path,
            format=format,
            dpi=dpi,
            bbox_inches='tight',
            **kwargs
        )

        return file_path
