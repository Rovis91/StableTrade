import os
import time
import logging
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import redis
from redis.exceptions import ConnectionError
import subprocess
from celery import Celery
from celery.result import AsyncResult
from celery.app.control import Inspect
from src.logger import setup_logger

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None
}

def _setup_logging(self) -> logging.Logger:
    """Setup logging configuration."""
    log_file = self.log_dir / 'worker_manager.log'
    return setup_logger('worker_manager', str(log_file))

@dataclass
class WorkerStatus:
    """Data class for worker status information."""
    worker_id: str
    status: str
    tasks_completed: int
    memory_usage: float
    cpu_usage: float
    uptime: float
    last_heartbeat: float

class WorkerManager:
    """Worker management system with monitoring and error handling."""

    def __init__(self, 
                 num_workers: int = 4,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 log_dir: Optional[str] = None):
        """
        Initialize the worker manager.
        
        Args:
            num_workers: Number of Celery workers to start
            redis_host: Redis server host
            redis_port: Redis server port
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            log_dir: Directory for log files
        """
        self.num_workers = num_workers
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.redis_client: Optional[redis.Redis] = None
        self.redis_process: Optional[subprocess.Popen] = None
        self.celery_app: Optional[Celery] = None
        self.worker_processes: Dict[str, subprocess.Popen] = {}
        
        # Register signal handlers
        self._register_signal_handlers()
        
        self.logger.info(f"Worker Manager initialized with {num_workers} workers")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(self.log_dir / 'worker_manager.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        self.logger.debug("Signal handlers registered")

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.cleanup()
        
    def start_redis(self) -> bool:
        """Start Redis server with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Try connecting to existing Redis
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    socket_connect_timeout=5,  # Increased timeout
                    decode_responses=True  # Add this for better string handling
                )
                self.redis_client.ping()
                self.logger.info("Connected to existing Redis server")
                return True
            except redis.ConnectionError as e:
                self.logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                time.sleep(self.retry_delay)
                continue
            except Exception as e:
                self.logger.error(f"Unexpected Redis error: {e}")
                return False
        
        self.logger.error("Failed to connect to Redis after all retries")
        return False

    def test_redis_connection(self) -> bool:
        """Test Redis connection and basic operations."""
        try:
            import redis
            client = redis.Redis(
                host='localhost',
                port=6379,
                socket_connect_timeout=5,
                decode_responses=True
            )
            
            # Test basic operations
            test_key = 'test_connection'
            test_value = 'it_works'
            
            client.set(test_key, test_value)
            result = client.get(test_key)
            client.delete(test_key)
            
            if result == test_value:
                print("Redis connection test successful")
                return True
            else:
                print("Redis connection test failed: unexpected value")
                return False
                
        except Exception as e:
            print(f"Redis connection test failed: {e}")
            return False

    def initialize_celery(self) -> bool:
        """
        Initialize Celery application with configuration.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            broker_url = f'redis://{self.redis_host}:{self.redis_port}/0'
            backend_url = f'redis://{self.redis_host}:{self.redis_port}/1'
            
            self.celery_app = Celery(
                'optimization_tasks',
                broker=broker_url,
                backend=backend_url
            )
            
            # Configure Celery
            self.celery_app.conf.update({
                'worker_prefetch_multiplier': 1,
                'worker_max_tasks_per_child': 50,
                'worker_max_memory_per_child': 200000,
                'task_time_limit': 3600,
                'task_soft_time_limit': 3300,
                'task_track_started': True,
                'task_serializer': 'json',
                'result_serializer': 'json',
                'accept_content': ['json']
            })
            
            self.logger.info("Celery application initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Celery: {e}")
            return False

    def start_workers(self) -> bool:
        """Start Celery workers with monitoring."""
        try:
            self.logger.info(f"Starting {self.num_workers} Celery workers...")
            
            # Create tasks module path - make sure this points to your celery tasks file
            tasks_module = 'src.optimization.celery_tasks'
            
            for i in range(self.num_workers):
                worker_id = f"worker{i}"
                log_file = self.log_dir / f'worker_{worker_id}.log'
                
                # Modified command with absolute path and better worker options
                worker_process = subprocess.Popen(
                    [
                        'celery',
                        '-A', tasks_module,  # Use proper module path
                        'worker',
                        '--loglevel=INFO',
                        f'--hostname={worker_id}@%h',
                        '--pool=threads',  # Change to threads instead of solo
                        '--concurrency=1',
                        '--without-gossip',  # Reduce overhead
                        '--without-mingle',
                        '--without-heartbeat'
                    ],
                    env=dict(os.environ, PYTHONPATH=str(Path.cwd())),  # Add current directory to PYTHONPATH
                    stdout=open(log_file, 'a'),
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd())  # Set working directory
                )
                
                self.worker_processes[worker_id] = worker_process

            # Wait for workers to start and verify
            time.sleep(10)  # Give more time for workers to start
            
            # Check if processes are still running
            alive_workers = sum(1 for p in self.worker_processes.values() if p.poll() is None)
            
            if alive_workers == self.num_workers:
                self.logger.info(f"All {alive_workers} workers started successfully")
                return True
            else:
                self.logger.error(f"Only {alive_workers}/{self.num_workers} workers started")
                # Log worker output for debugging
                for worker_id, process in self.worker_processes.items():
                    log_file = self.log_dir / f'worker_{worker_id}.log'
                    if log_file.exists():
                        self.logger.error(f"Worker {worker_id} log:")
                        with open(log_file, 'r') as f:
                            self.logger.error(f.read())
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting workers: {e}")
            return False

    def get_worker_status(self) -> Dict[str, WorkerStatus]:
        """Get status information for all workers."""
        try:
            if not self.celery_app:
                return {}
                
            inspector = self.celery_app.control.inspect()
            active = inspector.active() or {}
            stats = inspector.stats() or {}
            
            worker_status = {}
            
            for worker_name, tasks in active.items():
                if worker_name in stats:
                    worker_stats = stats[worker_name]
                    
                    status = WorkerStatus(
                        worker_id=worker_name,
                        status='running' if tasks else 'idle',
                        tasks_completed=worker_stats.get('total', 0),
                        memory_usage=worker_stats.get('process_metrics', {}).get('memory_rss', 0),
                        cpu_usage=worker_stats.get('process_metrics', {}).get('cpu_percent', 0),
                        uptime=worker_stats.get('uptime', 0),
                        last_heartbeat=time.time()
                    )
                    
                    worker_status[worker_name] = status
            
            return worker_status
            
        except Exception as e:
            self.logger.error(f"Error getting worker status: {e}")
            return {}

    def monitor_task(self, task_id: str) -> Dict[str, Any]:
        """Monitor progress of a specific task."""
        try:
            if not self.celery_app:
                return {'status': 'error', 'message': 'Celery not initialized'}
                
            result = AsyncResult(task_id, app=self.celery_app)
            
            return {
                'id': task_id,
                'status': result.status,
                'successful': result.successful(),
                'failed': result.failed(),
                'runtime': result.runtime if result.ready() else None,
                'result': result.result if result.ready() else None,
                'traceback': result.traceback if result.failed() else None
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring task {task_id}: {e}")
            return {'status': 'error', 'message': str(e)}

    def cleanup(self) -> None:
        """Clean up all processes and connections."""
        self.logger.info("Starting cleanup...")
        
        # Stop workers
        for worker_id, process in self.worker_processes.items():
            try:
                self.logger.info(f"Stopping worker {worker_id}")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Worker {worker_id} did not stop gracefully, forcing...")
                process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping worker {worker_id}: {e}")
        
        # Stop Redis
        if self.redis_process:
            try:
                self.logger.info("Stopping Redis server")
                self.redis_process.terminate()
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Redis did not stop gracefully, forcing...")
                self.redis_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping Redis: {e}")
        
        # Close Redis client
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                self.logger.error(f"Error closing Redis client: {e}")
        
        self.logger.info("Cleanup completed")

    def __enter__(self):
        """Context manager entry."""
        if not self.start_redis():
            raise RuntimeError("Failed to start Redis")
            
        if not self.test_redis_connection():  # Add connection test
            self.cleanup()
            raise RuntimeError("Redis connection test failed")
            
        if not self.initialize_celery():
            self.cleanup()
            raise RuntimeError("Failed to initialize Celery")
            
        if not self.start_workers():
            self.cleanup()
            raise RuntimeError("Failed to start workers")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


if __name__ == "__main__":
    # Example usage
    worker_manager = WorkerManager(num_workers=4)
    worker_manager.test_redis_connection()