"""Main module for running strategy optimization and analysis."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
from src.optimization.visualization import (
    StrategyVisualizer,
    VisualizationConfig,
    ReportGenerator
)
from src.optimization.optimizer import StrategyOptimizer
from src.optimization.result_manager import OptimizationResults
from src.strategy.depeg_strategy import DepegStrategy

# At the start of your application
from src.logger import set_log_config, set_log_levels, setup_logger

# Configure logging
set_log_config(
    save_logs=False, 
    max_log_files=10, 
    log_dir='logs' 
)

# Set log levels for components
# Set more verbose logging levels
set_log_levels({
    'main': 'INFO',
    'optimizer': 'INFO',
    'worker_manager': 'INFO',
    'task_coordinator': 'INFO',
    'optimization_task': 'INFO', 
    'backtest_engine': 'INFO'
})

# Get logger for component
logger = setup_logger('main')
# Default configurations
DEFAULT_PARAMS = {
    'depeg_threshold': {'start': 0.01, 'end': 0.05, 'step': 0.01},
    'trade_amount': {'start': 0.1, 'end': 0.1, 'step': 0.1},
    'stop_loss': {'start': 0.01, 'end': 0.01, 'step': 0.01},
    'take_profit': {'start': 0.02, 'end': 0.10, 'step': 0.02},
    'trailing_stop': {'start': 0.01, 'end': 0.01, 'step': 0.01}
}

DEFAULT_CONFIG = {
    'asset': 'EUTEUR',
    'data_path': 'D:\\StableTrade_dataset\\EUTEUR_1m\\EUTEUR_1m_final_merged.csv',
    'initial_cash': 10000,
    'base_currency': 'EUR'
}

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    try:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        return input(f"{prompt}: ").strip() or default
    except KeyboardInterrupt:
        logger.info("User interrupted input")
        raise
    except Exception as e:
        logger.error(f"Error getting user input: {str(e)}")
        raise

def run_optimization(
    param_ranges: Dict[str, Dict[str, float]],
    base_config: Dict[str, Any],
    output_dir: Union[str, Path],
    strategy_class=DepegStrategy  
) -> str:
    """Run optimization process and save results."""
    try:
        logger.info("Starting optimization process...")
        
        # Validate inputs
        if not param_ranges or not base_config:
            raise ValueError("Invalid parameter ranges or base configuration")
            
        optimizer = StrategyOptimizer(
            strategy_class=strategy_class,  # Add strategy class
            param_ranges=param_ranges,
            base_config=base_config,
            output_dir=output_dir
        )
        
        results = optimizer.run_optimization()
        results_path = Path(output_dir) / 'optimization_results.csv'
        results.to_csv(results_path)
        
        logger.info(f"Optimization results saved to: {results_path}")
        return str(results_path)
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise

def analyze_results(
    results_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    visualization_config: Optional[VisualizationConfig] = None
) -> Dict[str, str]:
    """Analyze existing optimization results and generate visualizations."""
    try:
        logger.info(f"Starting analysis of results from: {results_path}")
        
        # Setup directories
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
            
        output_dir = Path(output_dir or results_path.parent / 'analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots_dir = output_dir / 'plots'
        reports_dir = output_dir / 'reports'
        plots_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)
        
        # Load and validate results
        results = pd.read_csv(results_path)
        if results.empty:
            raise ValueError("Empty results file")
            
        optimization_results = OptimizationResults()
        optimization_results.results = results
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = StrategyVisualizer(
            results_manager=optimization_results,
            output_dir=str(plots_dir),
            config=visualization_config
        )
        plot_paths = visualizer.generate_complete_analysis()
        
        # Generate report
        logger.info("Generating report...")
        reporter = ReportGenerator(
            results_manager=optimization_results,
            output_dir=output_dir
        )
        report_path = reporter.generate_markdown_report()
        
        return {
            'plots': plot_paths,
            'report': str(report_path)
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

def get_optimization_params(custom: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Get optimization parameters from user or use defaults."""
    try:
        if not custom:
            return DEFAULT_PARAMS.copy(), DEFAULT_CONFIG.copy()
            
        logger.info("Getting custom optimization parameters...")
        param_ranges = {}
        
        # Get parameter ranges
        for param, defaults in DEFAULT_PARAMS.items():
            param_ranges[param] = {
                'start': float(get_user_input(f"{param} start", str(defaults['start']))),
                'end': float(get_user_input(f"{param} end", str(defaults['end']))),
                'step': float(get_user_input(f"{param} step", str(defaults['step'])))
            }
            
        # Get base configuration
        base_config = {
            'asset': get_user_input("Asset name", DEFAULT_CONFIG['asset']),
            'data_path': get_user_input("Data file path", DEFAULT_CONFIG['data_path']),
            'initial_cash': float(get_user_input("Initial cash", str(DEFAULT_CONFIG['initial_cash']))),
            'base_currency': get_user_input("Base currency", DEFAULT_CONFIG['base_currency'])
        }
        
        return param_ranges, base_config
        
    except Exception as e:
        logger.error(f"Error getting optimization parameters: {str(e)}")
        raise

def show_menu() -> str:
    """Display main menu and get user choice."""
    menu_options = {
        '1': 'Lancer une nouvelle optimisation',
        '2': 'Analyser des résultats existants',
        '3': 'Optimisation et analyse',
        '4': 'Quitter'
    }
    
    while True:
        print("\n=== Menu Principal ===")
        for key, value in menu_options.items():
            print(f"{key}. {value}")
        
        try:
            choice = input("\nEntrez votre choix (1-4): ").strip()
            if choice in menu_options:
                return choice
            print("Choix invalide. Veuillez réessayer.")
        except KeyboardInterrupt:
            return '4'
        except Exception as e:
            logger.error(f"Error in menu selection: {str(e)}")
            print("Une erreur est survenue. Veuillez réessayer.")

def main() -> None:
    """Main function with interactive menu."""
    logger.info("Starting optimization tool...")
    
    while True:
        try:
            choice = show_menu()
            
            if choice == '4':
                print("\nAu revoir!")
                break
                
            if choice in ['1', '3']:
                # Handle optimization
                print("\n=== Configuration ===")
                config_choice = get_user_input(
                    "1. Utiliser les paramètres par défaut\n2. Configurer manuellement\nChoix",
                    "1"
                )
                
                param_ranges, base_config = get_optimization_params(config_choice == "2")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = get_user_input(
                    "Dossier de sortie",
                    f"optimization_results_{timestamp}"
                )
                
                print("\nLancement de l'optimisation...")
                results_path = run_optimization(param_ranges, base_config, output_dir)
                print(f"Résultats d'optimisation sauvegardés dans: {results_path}")
                
                if choice == '1':
                    continue
            
            if choice in ['2', '3']:
                # Handle analysis
                if choice == '2':
                    results_path = get_user_input(
                        "Chemin vers les résultats",
                        "./optimization_results/optimization_results.csv"
                    )
                
                vis_config = VisualizationConfig()
                if get_user_input("Configurer la visualisation? (o/N)", "n").lower() == 'o':
                    vis_config = VisualizationConfig(
                        figsize=(
                            int(get_user_input("Largeur du graphique", "15")),
                            int(get_user_input("Hauteur du graphique", "10"))
                        ),
                        dpi=int(get_user_input("DPI", "300")),
                        export_format=get_user_input("Format d'export", "svg")
                    )
                
                print("\nGénération des visualisations et rapports...")
                analysis_results = analyze_results(
                    results_path=results_path,
                    visualization_config=vis_config
                )
                
                print("\nRésultats de l'analyse:")
                print(f"Rapport: {analysis_results['report']}")
                print("\nGraphiques générés:")
                for plot_name, plot_path in analysis_results['plots'].items():
                    print(f"- {plot_name}: {plot_path}")
            
        except KeyboardInterrupt:
            print("\n\nOpération annulée par l'utilisateur.")
            continue
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"\nErreur: {str(e)}")
        
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()