"""Authentication module."""
from .config import auth_config
from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_current_verified_user,
    get_current_superuser,
    get_current_user_optional,
    CurrentUser,
    CurrentActiveUser,
    CurrentVerifiedUser,
    CurrentSuperuser,
    OptionalUser,
    RequirePermission,
    RequireRole,
    require_read,
    require_write,
    require_delete,
    require_admin,
    require_api_user
)
from .models import (
    User,
    Role,
    APIKey,
    UserSession,
    UserCreate,
    UserUpdate,
    UserResponse,
    Token,
    TokenData,
    LoginRequest,
    APIKeyCreate,
    APIKeyResponse
)
from .services import (
    UserService,
    SessionService,
    APIKeyService
)
from .routes import router as auth_router

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