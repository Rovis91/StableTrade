"""Base visualization functionality for strategy analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import warnings
from .config import VisualizationConfig

class BaseVisualizer:
    """Base class for visualization functionality.
    
    This class provides common functionality for all visualizers, including:
    - Plot style configuration
    - Directory management
    - Plot saving utilities
    
    Attributes:
        config (VisualizationConfig): Visualization configuration
        output_dir (Path): Directory for saving outputs
        plots_dir (Path): Directory for saving plots
        logger (logging.Logger): Logger instance
    """
    
    def __init__(self, 
                 output_dir: Union[str, Path], 
                 config: Optional[VisualizationConfig] = None):
        """Initialize the base visualizer.
        
        Args:
            output_dir: Directory path for saving visualizations
            config: Optional visualization configuration
            
        Raises:
            ValueError: If output_dir is invalid
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or VisualizationConfig()
        
        # Setup output directory
        try:
            self.output_dir = Path(output_dir)
            self.plots_dir = self.output_dir
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {self.plots_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {str(e)}")
            raise ValueError(f"Invalid output directory: {output_dir}") from e
        
        # Setup plotting style
        self._setup_plotting_style()
        
        # Suppress specific matplotlib warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn plotting styles."""
        try:
            # Set style
            plt.style.use(self.config.style)
            
            # Configure seaborn
            sns.set_palette(self.config.color_palette, n_colors=self.config.n_colors)
            sns.set_context("notebook", font_scale=self.config.font_scale)
            
            # Update matplotlib parameters
            plt.rcParams.update({
                'figure.figsize': self.config.figsize,
                'figure.dpi': self.config.dpi,
                'savefig.bbox': 'tight',
                'savefig.format': self.config.export_format,
                # Additional parameters for better plots
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 10,
                'figure.titlesize': 14,
                'figure.autolayout': True,
                'axes.grid': True,
                'grid.alpha': 0.3
            })
            
            self.logger.debug("Plot style configuration completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup plotting style: {str(e)}")
            raise
    
    def save_plot(self, name: str, **kwargs) -> Path:
        """Save the current plot with the given name.
        
        Args:
            name: Name for the plot file (without extension)
            **kwargs: Additional arguments to pass to plt.savefig()
            
        Returns:
            Path to the saved plot file
            
        Raises:
            IOError: If saving the plot fails
        """
        try:
            # Create output path
            output_path = self.plots_dir / f'{name}.{self.config.export_format}'
            
            # Default kwargs for saving
            save_kwargs = {
                'bbox_inches': 'tight',
                'dpi': self.config.dpi,
                'format': self.config.export_format,
                'facecolor': 'white',
                'edgecolor': 'none',
                'transparent': False
            }
            # Update with user provided kwargs
            save_kwargs.update(kwargs)
            
            # Save the plot
            plt.savefig(output_path, **save_kwargs)
            plt.close()
            
            self.logger.info(f"Saved plot to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save plot {name}: {str(e)}")
            raise IOError(f"Failed to save plot {name}") from e
    
    def get_plot_defaults(self) -> Dict[str, Any]:
        """Get default plotting parameters."""
        return {
            'figsize': self.config.figsize,
            'dpi': self.config.dpi,
            'style': self.config.style,
            'color_palette': self.config.color_palette,
            'n_colors': self.config.n_colors,
            'font_scale': self.config.font_scale,
            'show_annotations': self.config.show_annotations
        }