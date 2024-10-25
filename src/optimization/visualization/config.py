"""Configuration settings for the visualization package."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

@dataclass
class VisualizationConfig:
    """Configuration settings for visualizations.
    
    Attributes:
        figsize (Tuple[int, int]): Figure size in inches (width, height)
        dpi (int): Dots per inch for figure resolution
        style (str): Matplotlib style to use
        color_palette (str): Color palette name
        n_colors (int): Number of colors in the palette
        font_scale (float): Scale factor for font sizes
        export_format (str): Format for saving figures (e.g., 'svg', 'png')
        show_annotations (bool): Whether to show annotations on plots
        output_dir (Optional[Path]): Directory for saving output files
    """
    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Style settings
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = "husl"
    n_colors: int = 8
    font_scale: float = 1.2
    
    # Output settings
    export_format: str = 'svg'
    show_annotations: bool = True
    output_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'figsize': self.figsize,
            'dpi': self.dpi,
            'style': self.style,
            'color_palette': self.color_palette,
            'n_colors': self.n_colors,
            'font_scale': self.font_scale,
            'export_format': self.export_format,
            'show_annotations': self.show_annotations,
            'output_dir': str(self.output_dir) if self.output_dir else None
        }