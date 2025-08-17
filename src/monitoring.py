"""
Monitoring and logging utilities for IndoBERT Document Customer Service
Performance tracking, usage analytics, and system monitoring
"""

import logging
import time
import psutil
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from functools import wraps
import traceback

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None

@dataclass
class RequestMetrics:
    """Request performance metrics"""
    timestamp: datetime
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    query_length: int
    response_length: int
    intent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: datetime
    inference_time_ms: float
    context_retrieval_time_ms: float
    generation_time_ms: float
    total_tokens_processed: int
    cache_hit_rate: float
    memory_usage_mb: float

class MetricsCollector:
    """Collect and store various metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.request_metrics = deque(maxlen=max_history)
        self.model_metrics = deque(maxlen=max_history)
        self.error_log = deque(maxlen=max_history)
        
        # Thread-safe locks
        self.system_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.error_lock = threading.Lock()
        
        # Start system monitoring
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent
        )
        
        # GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                metrics.gpu_memory_used = gpu_memory_used
                metrics.gpu_memory_total = gpu_memory_total
        except ImportError:
            pass
        
        return metrics
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics to collection"""
        with self.system_lock:
            self.system_metrics.append(metrics)
    
    def add_request_metrics(self, metrics: RequestMetrics):
        """Add request metrics to collection"""
        with self.request_lock:
            self.request_metrics.append(metrics)
    
    def add_model_metrics(self, metrics: ModelMetrics):
        """Add model metrics to collection"""
        with self.model_lock:
            self.model_metrics.append(metrics)
    
    def add_error(self, error: str, context: Dict[str, Any]):
        """Add error to log"""
        with self.error_lock:
            self.error_log.append({
                'timestamp': datetime.now(),
                'error': error,
                'context': context,
                'traceback': traceback.format_exc()
            })
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics by time
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_requests = [m for m in self.request_metrics if m.timestamp >= cutoff_time]
        recent_model = [m for m in self.model_metrics if m.timestamp >= cutoff_time]
        recent_errors = [e for e in self.error_log if e['timestamp'] >= cutoff_time]
        
        summary = {
            'time_window_hours': hours,
            'timestamp': datetime.now(),
            'system': self._summarize_system_metrics(recent_system),
            'requests': self._summarize_request_metrics(recent_requests),
            'model': self._summarize_model_metrics(recent_model),
            'errors': {
                'total_errors': len(recent_errors),
                'error_rate': len(recent_errors) / max(len(recent_requests), 1),
                'top_errors': self._get_top_errors(recent_errors)
            }
        }
        
        return summary
    
    def _summarize_system_metrics(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Summarize system metrics"""
        if not metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        
        return {
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'current_used_gb': metrics[-1].memory_used_gb if metrics else 0
            },
            'disk_usage_percent': metrics[-1].disk_usage_percent if metrics else 0,
            'gpu_memory_used_gb': metrics[-1].gpu_memory_used if metrics and metrics[-1].gpu_memory_used else 0
        }
    
    def _summarize_request_metrics(self, metrics: List[RequestMetrics]) -> Dict[str, Any]:
        """Summarize request metrics"""
        if not metrics:
            return {}
        
        response_times = [m.response_time_ms for m in metrics]
        successful_requests = [m for m in metrics if m.success]
        
        # Group by endpoint
        by_endpoint = defaultdict(list)
        for m in metrics:
            by_endpoint[m.endpoint].append(m)
        
        return {
            'total_requests': len(metrics),
            'successful_requests': len(successful_requests),
            'success_rate': len(successful_requests) / len(metrics),
            'avg_response_time_ms': sum(response_times) / len(response_times),
            'p95_response_time_ms': self._percentile(response_times, 95),
            'p99_response_time_ms': self._percentile(response_times, 99),
            'requests_per_minute': len(metrics) / max(1, len(set(m.timestamp.strftime('%Y-%m-%d %H:%M') for m in metrics))),
            'by_endpoint': {
                endpoint: {
                    'count': len(reqs),
                    'avg_response_time': sum(r.response_time_ms for r in reqs) / len(reqs),
                    'success_rate': sum(1 for r in reqs if r.success) / len(reqs)
                }
                for endpoint, reqs in by_endpoint.items()
            }
        }
    
    def _summarize_model_metrics(self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Summarize model metrics"""
        if not metrics:
            return {}
        
        inference_times = [m.inference_time_ms for m in metrics]
        total_tokens = sum(m.total_tokens_processed for m in metrics)
        
        return {
            'total_inferences': len(metrics),
            'avg_inference_time_ms': sum(inference_times) / len(inference_times),
            'total_tokens_processed': total_tokens,
            'tokens_per_second': total_tokens / max(1, sum(inference_times) / 1000),
            'avg_cache_hit_rate': sum(m.cache_hit_rate for m in metrics) / len(metrics),
            'avg_memory_usage_mb': sum(m.memory_usage_mb for m in metrics) / len(metrics)
        }
    
    def _get_top_errors(self, errors: List[Dict]) -> List[Dict]:
        """Get top error types"""
        error_counts = defaultdict(int)
        for error in errors:
            error_type = error['error']
            error_counts[error_type] += 1
        
        return [
            {'error': error, 'count': count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(percentile / 100 * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _monitor_system(self):
        """Background system monitoring"""
        while self.monitoring_active:
            try:
                metrics = self.collect_system_metrics()
                self.add_system_metrics(metrics)
                time.sleep(60)  # Collect every minute
            except Exception as e:
                logging.error(f"System monitoring error: {e}")
                time.sleep(60)
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
    
    def export_metrics(self, file_path: str):
        """Export all metrics to JSON file"""
        data = {
            'exported_at': datetime.now().isoformat(),
            'system_metrics': [asdict(m) for m in self.system_metrics],
            'request_metrics': [asdict(m) for m in self.request_metrics],
            'model_metrics': [asdict(m) for m in self.model_metrics],
            'errors': list(self.error_log)
        }
        
        # Convert datetime objects to strings
        data = self._serialize_datetime(data)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Metrics exported to {file_path}")
    
    def _serialize_datetime(self, obj):
        """Recursively serialize datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        else:
            return obj

# Global metrics collector instance
metrics_collector = MetricsCollector()

def monitor_request(func):
    """Decorator to monitor API requests"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        request = args[0] if args else None
        
        try:
            result = await func(*args, **kwargs)
            
            # Extract metrics
            response_time = (time.time() - start_time) * 1000
            endpoint = getattr(request, 'url', {}).get('path', 'unknown') if request else 'unknown'
            
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                endpoint=endpoint,
                method=getattr(request, 'method', 'unknown') if request else 'unknown',
                response_time_ms=response_time,
                status_code=200,  # Assume success if no exception
                query_length=0,   # Would need to extract from request
                response_length=len(str(result)) if result else 0,
                success=True
            )
            
            metrics_collector.add_request_metrics(metrics)
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            endpoint = getattr(request, 'url', {}).get('path', 'unknown') if request else 'unknown'
            
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                endpoint=endpoint,
                method=getattr(request, 'method', 'unknown') if request else 'unknown',
                response_time_ms=response_time,
                status_code=500,
                query_length=0,
                response_length=0,
                success=False,
                error_message=str(e)
            )
            
            metrics_collector.add_request_metrics(metrics)
            metrics_collector.add_error(str(e), {'endpoint': endpoint, 'args': str(args)})
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            response_time = (time.time() - start_time) * 1000
            
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                endpoint=func.__name__,
                method='CALL',
                response_time_ms=response_time,
                status_code=200,
                query_length=0,
                response_length=len(str(result)) if result else 0,
                success=True
            )
            
            metrics_collector.add_request_metrics(metrics)
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                endpoint=func.__name__,
                method='CALL',
                response_time_ms=response_time,
                status_code=500,
                query_length=0,
                response_length=0,
                success=False,
                error_message=str(e)
            )
            
            metrics_collector.add_request_metrics(metrics)
            metrics_collector.add_error(str(e), {'function': func.__name__, 'args': str(args)})
            raise
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def monitor_model_inference(func):
    """Decorator to monitor model inference"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            inference_time = (time.time() - start_time) * 1000
            
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                inference_time_ms=inference_time,
                context_retrieval_time_ms=0,  # Would need to measure separately
                generation_time_ms=inference_time,  # Approximation
                total_tokens_processed=0,  # Would need to extract from result
                cache_hit_rate=0.0,  # Would need to track cache usage
                memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024
            )
            
            metrics_collector.add_model_metrics(metrics)
            return result
            
        except Exception as e:
            metrics_collector.add_error(str(e), {'function': func.__name__})
            raise
    
    return wrapper

class PerformanceLogger:
    """Logger for performance analysis"""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('performance')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_request(self, endpoint: str, response_time: float, success: bool, **kwargs):
        """Log request performance"""
        self.logger.info(
            f"REQUEST | {endpoint} | {response_time:.2f}ms | {'SUCCESS' if success else 'FAILED'} | {kwargs}"
        )
    
    def log_model_inference(self, inference_time: float, tokens: int, **kwargs):
        """Log model inference performance"""
        tokens_per_second = tokens / (inference_time / 1000) if inference_time > 0 else 0
        self.logger.info(
            f"INFERENCE | {inference_time:.2f}ms | {tokens} tokens | {tokens_per_second:.1f} tokens/s | {kwargs}"
        )
    
    def log_system_alert(self, metric: str, value: float, threshold: float):
        """Log system performance alerts"""
        self.logger.warning(
            f"ALERT | {metric} | {value:.2f} > {threshold:.2f} threshold"
        )

# Global performance logger
performance_logger = PerformanceLogger()

def get_health_status() -> Dict[str, Any]:
    """Get current system health status"""
    try:
        metrics = metrics_collector.collect_system_metrics()
        summary = metrics_collector.get_metrics_summary(hours=1)
        
        # Determine health status based on thresholds
        health_status = "healthy"
        alerts = []
        
        # CPU threshold
        if metrics.cpu_percent > 80:
            health_status = "warning"
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory threshold
        if metrics.memory_percent > 85:
            health_status = "critical" if metrics.memory_percent > 95 else "warning"
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Error rate threshold
        error_rate = summary.get('errors', {}).get('error_rate', 0)
        if error_rate > 0.1:  # 10% error rate
            health_status = "warning"
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(metrics),
            "summary": summary,
            "alerts": alerts
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def export_monitoring_report(output_dir: str = "logs"):
    """Export comprehensive monitoring report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export metrics
    metrics_file = os.path.join(output_dir, f"metrics_{timestamp}.json")
    metrics_collector.export_metrics(metrics_file)
    
    # Generate summary report
    summary = metrics_collector.get_metrics_summary(hours=24)
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Monitoring report exported to {output_dir}")
    
    return {
        "metrics_file": metrics_file,
        "summary_file": summary_file,
        "timestamp": timestamp
    }

# Cleanup function
def cleanup_monitoring():
    """Cleanup monitoring resources"""
    metrics_collector.stop_monitoring()
    logging.info("Monitoring cleanup completed")
