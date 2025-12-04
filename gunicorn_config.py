"""
Gunicorn configuration for Speech Emotion Recognition API
Optimized for handling long-running requests with numba compilation
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker on free tier to save memory
worker_class = 'gthread'
threads = 2
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings - increased for numba JIT compilation
timeout = 120  # 2 minutes for first request with numba compilation
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'speech_emotion_api'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload app to trigger numba compilation during startup
preload_app = True

# Restart workers after this many requests to prevent memory leaks
max_requests = 100
max_requests_jitter = 10

def on_starting(server):
    """Called just before the master process is initialized."""
    print("[INFO] Gunicorn is starting...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("[INFO] Gunicorn is reloading...")

def when_ready(server):
    """Called just after the server is started."""
    print("[INFO] Gunicorn is ready. Spawning workers")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    print(f"[INFO] Worker {worker.pid} received INT/QUIT signal")

def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    print(f"[WARNING] Worker {worker.pid} received ABORT signal")
