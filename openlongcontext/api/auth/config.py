"""Authentication configuration."""
import os

from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class AuthConfig(BaseSettings):
    """Authentication configuration settings."""

    # JWT Settings
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production"),
        description="Secret key for JWT encoding"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration time in minutes")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration time in days")

    # API Key Settings
    api_key_header_name: str = Field(default="X-API-Key", description="Header name for API key")
    api_key_prefix: str = Field(default="olc_", description="Prefix for API keys")

    # Password Policy
    min_password_length: int = Field(default=8, description="Minimum password length")
    require_uppercase: bool = Field(default=True, description="Require uppercase letter in password")
    require_lowercase: bool = Field(default=True, description="Require lowercase letter in password")
    require_digits: bool = Field(default=True, description="Require digit in password")
    require_special_chars: bool = Field(default=True, description="Require special character in password")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_default: str = Field(default="100/minute", description="Default rate limit")
    rate_limit_auth: str = Field(default="5/minute", description="Rate limit for auth endpoints")
    rate_limit_api: str = Field(default="1000/hour", description="Rate limit for API endpoints")

    # Security Headers
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow credentials in CORS")
    cors_methods: list[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: list[str] = Field(default=["*"], description="Allowed CORS headers")

    # Session Settings
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    max_sessions_per_user: int = Field(default=5, description="Maximum concurrent sessions per user")

    # Account Security
    max_login_attempts: int = Field(default=5, description="Maximum login attempts before lockout")
    lockout_duration_minutes: int = Field(default=30, description="Account lockout duration in minutes")
    password_reset_token_expire_minutes: int = Field(default=15, description="Password reset token expiration")
    email_verification_required: bool = Field(default=True, description="Require email verification")

    # OAuth2 Settings (optional)
    oauth2_enabled: bool = Field(default=False, description="Enable OAuth2 authentication")
    oauth2_providers: list[str] = Field(default=[], description="Enabled OAuth2 providers")

    class Config:
        env_prefix = "AUTH_"
        case_sensitive = False


auth_config = AuthConfig()
