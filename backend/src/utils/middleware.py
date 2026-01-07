"""
Middleware for request/response logging and processing.
"""
import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log HTTP requests and responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware or route handler
            
        Returns:
            Response object
        """
        start_time = time.time()
        
        # Log request details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        
        # Log request (excluding sensitive paths)
        if not path.startswith("/docs") and not path.startswith("/redoc") and not path.startswith("/openapi.json"):
            logger.info(
                f"Incoming request: {method} {path}",
                extra={
                    "client_ip": client_ip,
                    "method": method,
                    "path": path,
                    "query_params": query_params,
                }
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response details
            status_code = response.status_code
            
            if not path.startswith("/docs") and not path.startswith("/redoc") and not path.startswith("/openapi.json"):
                logger.info(
                    f"Request completed: {method} {path}",
                    extra={
                        "status_code": status_code,
                        "process_time": f"{process_time:.3f}s",
                        "client_ip": client_ip,
                    }
                )
            
            # Add timing header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {method} {path}",
                extra={
                    "error": str(e),
                    "process_time": f"{process_time:.3f}s",
                    "client_ip": client_ip,
                },
                exc_info=True
            )
            
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.
        
        Args:
            request: Incoming request
            call_next: Next middleware or route handler
            
        Returns:
            Response object with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

