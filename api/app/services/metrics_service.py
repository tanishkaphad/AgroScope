"""AgroScope Metrics and Uptime Monitoring Service."""

from __future__ import annotations

import os
import psutil
import time
from typing import Dict, Any


class MetricsService:
    _instance = None

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.inference_count: int = 0

    @classmethod
    def get_instance(cls) -> MetricsService:
        if cls._instance is None:
            cls._instance = MetricsService()
        return cls._instance

    def increment_request(self) -> None:
        self.request_count += 1

    def increment_inference(self) -> None:
        self.inference_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        uptime = round(time.time() - self.start_time, 2)
        rss_mb = round(mem_info.rss / (1024 * 1024), 2)

        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "total_inferences": self.inference_count,
            "memory_rss_mb": rss_mb,
            "cpu_percent": process.cpu_percent(),
            "status": "HEALTHY",
        }


metrics_service = MetricsService.get_instance()
