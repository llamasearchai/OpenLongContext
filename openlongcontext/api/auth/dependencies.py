"""Authentication dependencies for FastAPI."""
from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from jose import JWTError
from sqlalchemy.orm import Session

from ..database import get_db
from .config import auth_config
from .jwt import decode_token
from .models import APIKey, TokenType, User, UserSession

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name=auth_config.api_key_header_name, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class PermissionChecker:
    """Permission dependency checker."""

    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions

    def __call__(self, current_user: User = Depends(lambda: get_current_user())):
        user_permissions = current_user.get_permissions()
        for permission in self.required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {permission}"
                )
        return current_user


class RoleChecker:
    """Role dependency checker."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(lambda: get_current_user())):
        if not any(current_user.has_role(role) for role in self.allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return current_user


async def get_current_user_optional(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from various auth methods (optional)."""
    # Try Bearer token first
    if bearer_token and bearer_token.credentials:
        token = bearer_token.credentials

    # Try JWT token
    if token:
        try:
            token_data = decode_token(token)
            if not token_data or token_data.token_type != TokenType.ACCESS:
                return None

            # Check if session exists and is valid
            session = db.query(UserSession).filter(
                UserSession.token == token,
                UserSession.expires_at > datetime.utcnow()
            ).first()

            if not session:
                return None

            # Update last activity
            session.last_activity = datetime.utcnow()
            db.commit()

            user = db.query(User).filter(User.id == token_data.user_id).first()
            if user and user.is_active:
                return user

        except JWTError:
            pass

    # Try API key
    if api_key:
        # Check if it's a valid API key format
        if not api_key.startswith(auth_config.api_key_prefix):
            return None

        api_key_obj = db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()

        if api_key_obj:
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                return None

            # Check IP whitelist
            if api_key_obj.allowed_ips:
                client_ip = request.client.host
                if client_ip not in api_key_obj.allowed_ips:
                    return None

            # Update last used
            api_key_obj.last_used = datetime.utcnow()
            db.commit()

            user = api_key_obj.user
            if user and user.is_active:
                # Set API key permissions in request state
                request.state.api_key_permissions = api_key_obj.permissions
                request.state.api_key_rate_limit = api_key_obj.rate_limit
                return user

    return None


async def get_current_user(
    user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Get current user (required)."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if account is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is temporarily locked due to too many failed login attempts"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current verified user."""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email address."
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# Convenience type annotations
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]
CurrentVerifiedUser = Annotated[User, Depends(get_current_verified_user)]
CurrentSuperuser = Annotated[User, Depends(get_current_superuser)]
OptionalUser = Annotated[Optional[User], Depends(get_current_user_optional)]


# Permission dependencies
RequirePermission = PermissionChecker
RequireRole = RoleChecker


# Common permission checks
require_read = RequirePermission(["read"])
require_write = RequirePermission(["write"])
require_delete = RequirePermission(["delete"])
require_admin = RequireRole(["admin"])
require_api_user = RequireRole(["api_user", "admin"])
