"""Authentication routes."""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .models import (
    UserCreate, UserResponse, UserUpdate,
    Token, LoginRequest, 
    PasswordResetRequest, PasswordResetConfirm, ChangePasswordRequest,
    APIKeyCreate, APIKeyResponse
)
from .dependencies import (
    CurrentUser, CurrentActiveUser, CurrentVerifiedUser, CurrentSuperuser,
    get_current_user_optional
)
from .services import UserService, SessionService, APIKeyService
from .jwt import create_email_verification_token
from .config import auth_config
from ..database import get_db


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    user_service = UserService(db)
    
    # Create user
    user = user_service.create_user(
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        full_name=user_data.full_name,
        is_verified=not auth_config.email_verification_required  # Auto-verify if not required
    )
    
    # Generate email verification token if required
    if auth_config.email_verification_required:
        verification_token = create_email_verification_token(user.email, user.id)
        # In production, send this via email
        # For now, return it in the response (remove in production!)
        return {
            **UserResponse.from_orm(user).dict(),
            "verification_token": verification_token  # REMOVE IN PRODUCTION
        }
    
    return UserResponse.from_orm(user)


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login with username/email and password."""
    user_service = UserService(db)
    session_service = SessionService(db)
    
    # Authenticate user
    user = user_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    # Create session
    access_token, refresh_token = session_service.create_session(
        user=user,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        remember_me=form_data.client_id == "remember"  # Hack to pass remember_me
    )
    
    # Set refresh token as HTTP-only cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,  # Use HTTPS in production
        samesite="lax",
        max_age=auth_config.refresh_token_expire_days * 24 * 60 * 60
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=auth_config.access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: Request,
    response: Response,
    refresh_token: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    # Get refresh token from cookie if not provided
    if not refresh_token:
        refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required"
        )
    
    session_service = SessionService(db)
    tokens = session_service.refresh_session(refresh_token)
    
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    access_token, new_refresh_token = tokens
    
    # Update refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=auth_config.refresh_token_expire_days * 24 * 60 * 60
    )
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=auth_config.access_token_expire_minutes * 60
    )


@router.post("/logout")
async def logout(
    response: Response,
    current_user: CurrentUser,
    authorization: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Logout current user."""
    session_service = SessionService(db)
    
    # Revoke current session
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        session_service.revoke_session(token)
    
    # Clear refresh token cookie
    response.delete_cookie("refresh_token")
    
    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all(
    current_user: CurrentUser,
    db: Session = Depends(get_db)
):
    """Logout from all sessions."""
    session_service = SessionService(db)
    count = session_service.revoke_all_sessions(current_user.id)
    
    return {"message": f"Revoked {count} sessions"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: CurrentActiveUser
):
    """Get current user profile."""
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: CurrentActiveUser,
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    user_service = UserService(db)
    updated_user = user_service.update_user(current_user.id, user_update)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(updated_user)


@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: CurrentActiveUser,
    db: Session = Depends(get_db)
):
    """Change current user password."""
    # Verify current password
    if not current_user.verify_password(password_data.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    user_service = UserService(db)
    user_service.update_user(
        current_user.id,
        UserUpdate(password=password_data.new_password)
    )
    
    # Revoke all sessions except current
    session_service = SessionService(db)
    # This is a simplified approach - in production, preserve current session
    
    return {"message": "Password changed successfully"}


@router.post("/verify-email/{token}")
async def verify_email(
    token: str,
    db: Session = Depends(get_db)
):
    """Verify email address with token."""
    user_service = UserService(db)
    
    if user_service.verify_email(token):
        return {"message": "Email verified successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )


@router.post("/resend-verification")
async def resend_verification(
    current_user: CurrentUser,
    db: Session = Depends(get_db)
):
    """Resend email verification."""
    if current_user.is_verified:
        return {"message": "Email already verified"}
    
    verification_token = create_email_verification_token(
        current_user.email, 
        current_user.id
    )
    
    # In production, send this via email
    return {
        "message": "Verification email sent",
        "token": verification_token  # REMOVE IN PRODUCTION
    }


@router.post("/forgot-password")
async def forgot_password(
    password_reset: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """Request password reset."""
    user_service = UserService(db)
    
    try:
        reset_token = user_service.request_password_reset(password_reset.email)
        # In production, send this via email
        return {
            "message": "If the email exists, a reset link has been sent.",
            "token": reset_token  # REMOVE IN PRODUCTION
        }
    except HTTPException as e:
        # Always return success to prevent email enumeration
        return {"message": "If the email exists, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(
    password_reset: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """Reset password with token."""
    user_service = UserService(db)
    
    if user_service.reset_password(password_reset.token, password_reset.new_password):
        return {"message": "Password reset successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )


# API Key Management
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: CurrentVerifiedUser,
    db: Session = Depends(get_db)
):
    """Create a new API key."""
    api_key_service = APIKeyService(db)
    
    api_key = api_key_service.create_api_key(
        user=current_user,
        name=api_key_data.name,
        expires_in_days=api_key_data.expires_in_days,
        permissions=api_key_data.permissions,
        rate_limit=api_key_data.rate_limit,
        allowed_ips=api_key_data.allowed_ips
    )
    
    return APIKeyResponse(
        id=api_key.id,
        key=api_key.key,  # Only shown once
        name=api_key.name,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at
    )


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: CurrentVerifiedUser,
    db: Session = Depends(get_db)
):
    """List user's API keys."""
    api_key_service = APIKeyService(db)
    api_keys = api_key_service.get_user_api_keys(current_user.id)
    
    # Don't include the actual key in list response
    return [
        APIKeyResponse(
            id=key.id,
            key="***",  # Hidden
            name=key.name,
            created_at=key.created_at,
            expires_at=key.expires_at
        )
        for key in api_keys
        if key.is_active
    ]


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: CurrentVerifiedUser,
    db: Session = Depends(get_db)
):
    """Delete an API key."""
    api_key_service = APIKeyService(db)
    
    if api_key_service.delete_api_key(key_id, current_user.id):
        return {"message": "API key deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )


# Admin endpoints
@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: CurrentSuperuser,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    from .models import User
    
    users = db.query(User).offset(skip).limit(limit).all()
    return [UserResponse.from_orm(user) for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: CurrentSuperuser,
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)."""
    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: CurrentSuperuser,
    db: Session = Depends(get_db)
):
    """Delete user (admin only)."""
    user_service = UserService(db)
    
    if user_service.delete_user(user_id):
        return {"message": "User deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )