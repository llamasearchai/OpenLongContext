"""Authentication module."""
from .config import auth_config
from .dependencies import (
    CurrentActiveUser,
    CurrentSuperuser,
    CurrentUser,
    CurrentVerifiedUser,
    OptionalUser,
    RequirePermission,
    RequireRole,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    get_current_user_optional,
    get_current_verified_user,
    require_admin,
    require_api_user,
    require_delete,
    require_read,
    require_write,
)
from .models import (
    APIKey,
    APIKeyCreate,
    APIKeyResponse,
    LoginRequest,
    Role,
    Token,
    TokenData,
    User,
    UserCreate,
    UserResponse,
    UserSession,
    UserUpdate,
)
from .routes import router as auth_router
from .services import APIKeyService, SessionService, UserService

__all__ = [
    # Config
    "auth_config",

    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "get_current_superuser",
    "get_current_user_optional",
    "CurrentUser",
    "CurrentActiveUser",
    "CurrentVerifiedUser",
    "CurrentSuperuser",
    "OptionalUser",
    "RequirePermission",
    "RequireRole",
    "require_read",
    "require_write",
    "require_delete",
    "require_admin",
    "require_api_user",

    # Models
    "User",
    "Role",
    "APIKey",
    "UserSession",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "Token",
    "TokenData",
    "LoginRequest",
    "APIKeyCreate",
    "APIKeyResponse",

    # Services
    "UserService",
    "SessionService",
    "APIKeyService",

    # Router
    "auth_router"
]
