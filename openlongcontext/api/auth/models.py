"""Authentication database models."""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, EmailStr, Field, validator
from passlib.context import CryptContext
import secrets
import string
from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Table, JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# SQLAlchemy Base
Base = declarative_base()

# Association table for many-to-many relationship between users and roles
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id')),
    Column('role_id', String, ForeignKey('roles.id'))
)


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    API_USER = "api_user"
    READONLY = "readonly"


class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"


# SQLAlchemy Models
class User(Base):
    """User database model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: f"user_{secrets.token_urlsafe(16)}")
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Security fields
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str):
        """Set password hash."""
        self.hashed_password = pwd_context.hash(password)
        self.password_changed_at = datetime.utcnow()
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)
    
    def get_permissions(self) -> set:
        """Get all permissions for the user."""
        permissions = set()
        for role in self.roles:
            permissions.update(role.permissions or [])
        return permissions


class Role(Base):
    """Role database model."""
    __tablename__ = "roles"
    
    id = Column(String, primary_key=True, default=lambda: f"role_{secrets.token_urlsafe(8)}")
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    permissions = Column(JSON, nullable=True)  # List of permission strings
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")


class APIKey(Base):
    """API Key database model."""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: f"apikey_{secrets.token_urlsafe(16)}")
    key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Permissions and rate limits
    permissions = Column(JSON, nullable=True)  # Override user permissions
    rate_limit = Column(String, nullable=True)  # Custom rate limit
    allowed_ips = Column(JSON, nullable=True)  # IP whitelist
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    @staticmethod
    def generate_key(prefix: str = "olc_") -> str:
        """Generate a secure API key."""
        return f"{prefix}{secrets.token_urlsafe(32)}"


class UserSession(Base):
    """User session database model."""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: f"session_{secrets.token_urlsafe(16)}")
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    refresh_token = Column(String, unique=True, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")


# Pydantic Models for API
class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    is_superuser: bool = False


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password complexity."""
        from .config import auth_config
        
        if auth_config.require_uppercase and not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if auth_config.require_lowercase and not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if auth_config.require_digits and not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if auth_config.require_special_chars and not any(c in string.punctuation for c in v):
            raise ValueError('Password must contain at least one special character')
        return v


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class UserInDB(UserBase):
    """User in database schema."""
    id: str
    hashed_password: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    roles: List[str] = []
    
    class Config:
        orm_mode = True


class UserResponse(UserBase):
    """User response schema."""
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    roles: List[str] = []
    
    class Config:
        orm_mode = True


class Token(BaseModel):
    """Token schema."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    scopes: List[str] = []
    token_type: TokenType = TokenType.ACCESS


class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str
    expires_in_days: Optional[int] = None
    permissions: Optional[List[str]] = None
    rate_limit: Optional[str] = None
    allowed_ips: Optional[List[str]] = None


class APIKeyResponse(BaseModel):
    """API key response schema."""
    id: str
    key: str  # Only returned on creation
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str  # Can be username or email
    password: str
    remember_me: bool = False


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str