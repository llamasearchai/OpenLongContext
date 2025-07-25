"""Security middleware for FastAPI."""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .auth.config import auth_config
from .auth.dependencies import get_current_user_optional
from .database import get_db

# Rate limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[auth_config.rate_limit_default] if auth_config.rate_limit_enabled else [],
    enabled=auth_config.rate_limit_enabled
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Remove server header
        response.headers.pop("Server", None)

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        print(f"[{request.state.request_id}] {request.method} {request.url.path}")

        response = await call_next(request)

        # Calculate request duration
        duration = time.time() - start_time

        # Log response
        print(f"[{request.state.request_id}] {response.status_code} - {duration:.3f}s")

        # Add timing header
        response.headers["X-Process-Time"] = str(duration)

        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Handle authentication and add user to request state."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for certain paths
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/auth/login", "/auth/register"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)

        # Try to get current user
        try:
            # Create a fake dependency injection context
            db = next(get_db())
            try:
                user = await get_current_user_optional(
                    request=request,
                    token=request.headers.get("Authorization", "").replace("Bearer ", ""),
                    api_key=request.headers.get(auth_config.api_key_header_name),
                    bearer_token=None,
                    db=db
                )
                request.state.user = user
            finally:
                db.close()
        except Exception:
            request.state.user = None

        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Custom rate limit middleware with API key support."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if user has custom rate limit (e.g., from API key)
        if hasattr(request.state, "api_key_rate_limit") and request.state.api_key_rate_limit:
            # Apply custom rate limit
            # This is a simplified implementation
            pass

        return await call_next(request)


def setup_middleware(app):
    """Setup all middleware for the FastAPI app."""

    # CORS middleware
    if auth_config.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=auth_config.cors_origins,
            allow_credentials=auth_config.cors_credentials,
            allow_methods=auth_config.cors_methods,
            allow_headers=auth_config.cors_headers,
        )

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Request ID
    app.add_middleware(RequestIDMiddleware)

    # Logging
    app.add_middleware(LoggingMiddleware)

    # Authentication
    app.add_middleware(AuthenticationMiddleware)

    # Rate limiting
    if auth_config.rate_limit_enabled:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)
        app.add_middleware(RateLimitMiddleware)


# Rate limit decorators for specific endpoints
auth_limit = limiter.limit(auth_config.rate_limit_auth)
api_limit = limiter.limit(auth_config.rate_limit_api)
