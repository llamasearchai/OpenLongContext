"""Authentication services."""
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import secrets
from .models import (
    User, Role, APIKey, UserSession,
    UserCreate, UserUpdate, APIKeyCreate,
    pwd_context
)
from .jwt import (
    create_access_token, create_refresh_token,
    create_password_reset_token, create_email_verification_token,
    verify_token, TokenType
)
from .config import auth_config


class UserService:
    """User management service."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        is_superuser: bool = False,
        is_verified: bool = False,
        roles: Optional[List[str]] = None
    ) -> User:
        """Create a new user."""
        # Check if user already exists
        if self.get_user_by_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        if self.get_user_by_username(username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create user
        user = User(
            email=email,
            username=username,
            full_name=full_name,
            is_superuser=is_superuser,
            is_verified=is_verified
        )
        user.set_password(password)
        
        # Add default role
        if not roles:
            roles = ["user"]
        
        # Add roles
        for role_name in roles:
            role = self.db.query(Role).filter(Role.name == role_name).first()
            if role:
                user.roles.append(role)
        
        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User creation failed"
            )
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        # Try to find user by username or email
        user = self.get_user_by_username(username)
        if not user:
            user = self.get_user_by_email(username)
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account locked until {user.locked_until}. Too many failed login attempts."
            )
        
        # Verify password
        if not user.verify_password(password):
            # Increment failed login attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= auth_config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=auth_config.lockout_duration_minutes
                )
            
            self.db.commit()
            return None
        
        # Reset failed login attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user
    
    def update_user(
        self,
        user_id: str,
        user_update: UserUpdate
    ) -> Optional[User]:
        """Update user information."""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        update_data = user_update.dict(exclude_unset=True)
        
        # Handle password update
        if "password" in update_data:
            user.set_password(update_data.pop("password"))
        
        # Update other fields
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        try:
            self.db.commit()
            self.db.refresh(user)
            return user
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Update failed. Email or username may already be in use."
            )
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        return True
    
    def verify_email(self, token: str) -> bool:
        """Verify user email with token."""
        payload = verify_token(token, TokenType.EMAIL_VERIFICATION)
        if not payload:
            return False
        
        user = self.get_user_by_id(payload.get("user_id"))
        if not user or user.email != payload.get("email"):
            return False
        
        user.is_verified = True
        self.db.commit()
        return True
    
    def request_password_reset(self, email: str) -> str:
        """Generate password reset token."""
        user = self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail="If the email exists, a reset link has been sent."
            )
        
        return create_password_reset_token(email)
    
    def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password with token."""
        payload = verify_token(token, TokenType.PASSWORD_RESET)
        if not payload:
            return False
        
        user = self.get_user_by_email(payload.get("email"))
        if not user:
            return False
        
        user.set_password(new_password)
        self.db.commit()
        return True


class SessionService:
    """Session management service."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False
    ) -> tuple[str, str]:
        """Create user session and return tokens."""
        # Check max sessions per user
        active_sessions = self.db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.expires_at > datetime.utcnow()
        ).count()
        
        if active_sessions >= auth_config.max_sessions_per_user:
            # Remove oldest session
            oldest_session = self.db.query(UserSession).filter(
                UserSession.user_id == user.id
            ).order_by(UserSession.created_at).first()
            if oldest_session:
                self.db.delete(oldest_session)
        
        # Create tokens
        access_token = create_access_token(
            data={
                "sub": user.id,
                "username": user.username,
                "scopes": list(user.get_permissions())
            }
        )
        
        refresh_token = create_refresh_token(
            data={
                "sub": user.id,
                "username": user.username
            },
            expires_delta=timedelta(days=30) if remember_me else None
        )
        
        # Create session
        session = UserSession(
            user_id=user.id,
            token=access_token,
            refresh_token=refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(
                minutes=auth_config.access_token_expire_minutes
            )
        )
        
        self.db.add(session)
        self.db.commit()
        
        return access_token, refresh_token
    
    def refresh_session(self, refresh_token: str) -> Optional[tuple[str, str]]:
        """Refresh session with refresh token."""
        # Find session
        session = self.db.query(UserSession).filter(
            UserSession.refresh_token == refresh_token
        ).first()
        
        if not session:
            return None
        
        # Verify refresh token
        payload = verify_token(refresh_token, TokenType.REFRESH)
        if not payload:
            return None
        
        user = session.user
        if not user or not user.is_active:
            return None
        
        # Create new tokens
        new_access_token = create_access_token(
            data={
                "sub": user.id,
                "username": user.username,
                "scopes": list(user.get_permissions())
            }
        )
        
        new_refresh_token = create_refresh_token(
            data={
                "sub": user.id,
                "username": user.username
            }
        )
        
        # Update session
        session.token = new_access_token
        session.refresh_token = new_refresh_token
        session.expires_at = datetime.utcnow() + timedelta(
            minutes=auth_config.access_token_expire_minutes
        )
        session.last_activity = datetime.utcnow()
        
        self.db.commit()
        
        return new_access_token, new_refresh_token
    
    def revoke_session(self, token: str) -> bool:
        """Revoke session by token."""
        session = self.db.query(UserSession).filter(
            UserSession.token == token
        ).first()
        
        if session:
            self.db.delete(session)
            self.db.commit()
            return True
        
        return False
    
    def revoke_all_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        count = self.db.query(UserSession).filter(
            UserSession.user_id == user_id
        ).delete()
        self.db.commit()
        return count


class APIKeyService:
    """API key management service."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_api_key(
        self,
        user: User,
        name: str,
        expires_in_days: Optional[int] = None,
        permissions: Optional[List[str]] = None,
        rate_limit: Optional[str] = None,
        allowed_ips: Optional[List[str]] = None
    ) -> APIKey:
        """Create API key for user."""
        api_key = APIKey(
            key=APIKey.generate_key(auth_config.api_key_prefix),
            name=name,
            user_id=user.id,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            permissions=permissions,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips
        )
        
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        
        return api_key
    
    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """Get all API keys for a user."""
        return self.db.query(APIKey).filter(
            APIKey.user_id == user_id
        ).all()
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke API key."""
        api_key = self.db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if api_key:
            api_key.is_active = False
            self.db.commit()
            return True
        
        return False
    
    def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Delete API key."""
        api_key = self.db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if api_key:
            self.db.delete(api_key)
            self.db.commit()
            return True
        
        return False