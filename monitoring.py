import time
import functools
import logging
from flask import request, g
from prometheus_client import Counter, Histogram, Gauge, Info

# Initialize logging
logger = logging.getLogger(__name__)

# Define Prometheus metrics
http_request_total = Counter(
    'http_request_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests',
)

app_info = Info(
    'mycology_pipeline_info',
    'Information about the mycology research pipeline'
)


def setup_metrics(app):
    """
    Setup metrics collection for a Flask app.
    
    Args:
        app: Flask application instance
    """
    # Set application info
    app_info.info({
        'version': '0.1.0',
        'name': 'mycology_research_pipeline'
    })
    
    # Register middleware to record all requests
    @app.before_request
    def before_request():
        g.start_time = time.time()
        active_requests.inc()
    
    @app.after_request
    def after_request(response):
        # Record request duration
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            endpoint = request.path
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
        
        # Record request count
        http_request_total.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
        ).inc()
        
        active_requests.dec()
        
        return response
    
    # Log setup completion
    logger.info("Prometheus metrics initialized")


def record_request_duration(func):
    """
    Decorator to record the duration of API function calls.
    
    Args:
        func: Function to wrap
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start_time
            
            # Get the function name as the endpoint
            endpoint = func.__name__
            
            # Record the duration
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
    
    return wrapper


def update_memory_usage():
    """Update memory usage metric."""
    import os
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_usage_bytes.set(process.memory_info().rss)


def start_metrics_collection_thread(interval=60):
    """
    Start a background thread to collect metrics periodically.
    
    Args:
        interval: Collection interval in seconds
    """
    import threading
    
    def collect_metrics():
        while True:
            try:
                update_memory_usage()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(interval)
    
    thread = threading.Thread(target=collect_metrics, daemon=True)
    thread.start()
    logger.info(f"Started metrics collection thread with interval {interval}s")
