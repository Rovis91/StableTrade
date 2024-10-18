import cProfile
import pstats
from pstats import SortKey
import os
import sys

def profile_main(script_name):
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Import the main script dynamically
    main_module = __import__(script_name)

    custom_log_levels = {
        'main': 'WARNING',
        'trade_manager': 'WARNING',
        'signal_database': 'WARNING',
        'depeg_strategy': 'WARNING',
        'portfolio': 'WARNING',
        'metrics': 'WARNING',
        'backtest_engine': 'WARNING'
    }

    # Run the profiler on the main function
    cProfile.runctx('main_module.main()', globals(), locals(), 'output.prof')

    # Print sorted stats
    with open('profiling_results.txt', 'w') as stream:
        p = pstats.Stats('output.prof', stream=stream)
        p.sort_stats(SortKey.CUMULATIVE).print_stats(30)

    print(f"Profiling completed. Results written to 'profiling_results.txt'")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        script_name = sys.argv[1]
    else:
        script_name = 'main_1_asset'  # default to main_1_asset if no argument is provided
    
    profile_main(script_name)