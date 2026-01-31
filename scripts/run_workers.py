#!/usr/bin/env python3
"""
RQ Worker Startup Script
Starts RQ workers for processing background jobs.

Usage:
    python scripts/run_workers.py                    # Start all queues
    python scripts/run_workers.py --queues generation refinement
    python scripts/run_workers.py --workers 4        # 4 worker processes
"""

import argparse
import logging
import os
import sys
import signal
from multiprocessing import Process
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis import Redis
from rq import Worker, Queue, Connection

from app.core.redis import get_redis, Queues, redis_health_check
from app.core.config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("rq.worker")


# All available queues in priority order
ALL_QUEUES = [
    Queues.HIGH_PRIORITY,
    Queues.GENERATION,
    Queues.REFINEMENT,
    Queues.EXTRACTION,
    Queues.DEFAULT,
    Queues.LOW_PRIORITY
]


def start_worker(queues: List[str], worker_name: str = None):
    """
    Start a single RQ worker.
    
    Args:
        queues: List of queue names to listen to
        worker_name: Optional worker identifier
    """
    redis_conn = get_redis()
    
    # Create Queue objects
    queue_objs = [Queue(name, connection=redis_conn) for name in queues]
    
    worker = Worker(
        queues=queue_objs,
        connection=redis_conn,
        name=worker_name,
        log_job_description=True,
        job_monitoring_interval=5
    )
    
    logger.info(f"Worker {worker_name or 'default'} starting on queues: {queues}")
    worker.work(with_scheduler=True)


def run_worker_process(queues: List[str], process_id: int):
    """Target function for worker processes."""
    worker_name = f"worker-{process_id}"
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"{worker_name}: Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    start_worker(queues, worker_name)


def main():
    parser = argparse.ArgumentParser(description="Start RQ workers for CineAI")
    parser.add_argument(
        "--queues", "-q",
        nargs="+",
        default=ALL_QUEUES,
        help="Queue names to listen to (default: all queues)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--burst", "-b",
        action="store_true",
        help="Run in burst mode (exit when queue is empty)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Redis connection and exit"
    )
    
    args = parser.parse_args()
    
    # Health check only
    if args.check:
        health = redis_health_check()
        print(f"Redis Status: {health}")
        sys.exit(0 if health.get("connected") else 1)
    
    # Verify Redis connection
    logger.info("Checking Redis connection...")
    health = redis_health_check()
    
    if not health.get("connected"):
        logger.error(f"Cannot connect to Redis: {health.get('error')}")
        logger.error(f"Redis URL: {health.get('url')}")
        sys.exit(1)
    
    logger.info(f"Redis connected: {health.get('redis_version')}")
    logger.info(f"Starting {args.workers} worker(s) on queues: {args.queues}")
    
    if args.workers == 1:
        # Single worker - run directly
        start_worker(args.queues, "worker-main")
    else:
        # Multiple workers - spawn processes
        processes: List[Process] = []
        
        def shutdown_all(signum, frame):
            logger.info("Shutting down all workers...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, shutdown_all)
        signal.signal(signal.SIGINT, shutdown_all)
        
        for i in range(args.workers):
            p = Process(
                target=run_worker_process,
                args=(args.queues, i + 1),
                name=f"worker-{i + 1}"
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker process {i + 1}/{args.workers} (PID: {p.pid})")
        
        # Wait for all workers
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
