import subprocess
import time
import logging
import atexit
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class WorkerManager:
    """Manages Redis server and Celery workers lifecycle."""
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize WorkerManager with default settings.

        Args:
            num_workers (int): Number of Celery workers to start.
        """
        self.num_workers = num_workers
        self.redis_process = None
        self.celery_process = None
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Register cleanup on program exit
        atexit.register(self.cleanup)

    def start_redis(self) -> bool:
        """
        Start Redis server if not already running.

        Returns:
            bool: True if Redis started successfully, False otherwise.
        """
        try:
            # Check if Redis is already running
            result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
            if result.stdout.strip() == 'PONG':
                logger.info("Redis server is already running")
                return True

            logger.info("Starting Redis server...")
            self.redis_process = subprocess.Popen(
                ['redis-server'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for Redis to start
            time.sleep(2)
            
            # Verify Redis is running
            result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
            if result.stdout.strip() == 'PONG':
                logger.info("Redis server started successfully")
                return True
            else:
                logger.error("Failed to start Redis server")
                return False

        except Exception as e:
            logger.error(f"Error starting Redis: {str(e)}")
            return False

    def start_celery_workers(self) -> bool:
        """
        Start Celery workers.

        Returns:
            bool: True if Celery workers started successfully, False otherwise.
        """
        try:
            logger.info(f"Starting {self.num_workers} Celery workers...")
            
            # Construct the celery command
            celery_module = 'src.optimization.celery_tasks'
            cmd = [
                'celery',
                '-A', celery_module,
                'worker',
                '--loglevel=INFO',
                f'--concurrency={self.num_workers}',
                '--pool=prefork',
                '--max-tasks-per-child=50'
            ]
            
            # Start celery worker process
            self.celery_process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for workers to start
            time.sleep(5)
            
            if self.celery_process.poll() is None:
                logger.info("Celery workers started successfully")
                return True
            else:
                logger.error("Failed to start Celery workers")
                return False

        except Exception as e:
            logger.error(f"Error starting Celery workers: {str(e)}")
            return False

    def stop_redis(self) -> None:
        """Stop Redis server."""
        try:
            if self.redis_process:
                self.redis_process.terminate()
                self.redis_process.wait(timeout=5)
                logger.info("Redis server stopped")
        except Exception as e:
            logger.error(f"Error stopping Redis: {str(e)}")

    def stop_celery_workers(self) -> None:
        """Stop Celery workers."""
        try:
            if self.celery_process:
                # Send SIGTERM to the main Celery process
                self.celery_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.celery_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.celery_process.kill()
                    self.celery_process.wait()
                
                logger.info("Celery workers stopped")
        except Exception as e:
            logger.error(f"Error stopping Celery workers: {str(e)}")

    def cleanup(self) -> None:
        """Clean up all processes on exit."""
        logger.info("Cleaning up worker processes...")
        self.stop_celery_workers()
        self.stop_redis()

    def __enter__(self):
        """Start services when entering context."""
        if not self.start_redis():
            raise RuntimeError("Failed to start Redis")
        if not self.start_celery_workers():
            self.stop_redis()
            raise RuntimeError("Failed to start Celery workers")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.cleanup()
