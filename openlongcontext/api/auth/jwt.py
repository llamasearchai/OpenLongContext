"""JWT token utilities."""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from .config import auth_config
from .models import TokenType, TokenData


def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=auth_config.access_token_expire_minutes)
    
    to_encode.update({
        "exp": expire,
        "type": TokenType.ACCESS.value,
        "iat": datetime.utcnow()
    })
    
    encoded_jwt = jwt.encode(
        to_encode, 
        auth_config.secret_key, 
        algorithm=auth_config.algorithm
    )
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=auth_config.refresh_token_expire_days)
    
    to_encode.update({
        "exp": expire,
        "type": TokenType.REFRESH.value,
        "iat": datetime.utcnow()
    })
    
    encoded_jwt = jwt.encode(
        to_encode, 
        auth_config.secret_key, 
        algorithm=auth_config.algorithm
    )
    return encoded_jwt


def create_password_reset_token(email: str) -> str:
    """Create password reset token."""
    expire = datetime.utcnow() + timedelta(minutes=auth_config.password_reset_token_expire_minutes)
    to_encode = {
        "email": email,
        "exp": expire,
        "type": TokenType.PASSWORD_RESET.value,
        "iat": datetime.utcnow()
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        auth_config.secret_key, 
        algorithm=auth_config.algorithm
    )
    return encoded_jwt


def create_email_verification_token(email: str, user_id: str) -> str:
    """Create email verification token."""
    expire = datetime.utcnow() + timedelta(days=7)  # 7 days to verify email
    to_encode = {
        "email": email,
        "user_id": user_id,
        "exp": expire,
        "type": TokenType.EMAIL_VERIFICATION.value,
        "iat": datetime.utcnow()
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        auth_config.secret_key, 
        algorithm=auth_config.algorithm
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(
            token, 
            auth_config.secret_key, 
            algorithms=[auth_config.algorithm]
        )
        
        # Extract token type
        token_type = TokenType(payload.get("type", TokenType.ACCESS.value))
        
        # Extract user data based on token type
        if token_type in [TokenType.ACCESS, TokenType.REFRESH]:
            user_id = payload.get("sub")
            username = payload.get("username")
            scopes = payload.get("scopes", [])
            
            if user_id is None:
                return None
                
            return TokenData(
                user_id=user_id,
                username=username,
                scopes=scopes,
                token_type=token_type
            )
        
        return None
        
    except JWTError:
        return None


def verify_token(token: str, token_type: TokenType) -> Optional[Dict[str, Any]]:
    """Verify token and return payload if valid."""
    try:
        payload = jwt.decode(
            token, 
            auth_config.secret_key, 
            algorithms=[auth_config.algorithm]
        )
        
        # Verify token type
        if payload.get("type") != token_type.value:
            return None
            
        return payload
        
    except JWTError:
        return None


def create_api_token(user_id: str, scopes: list[str]) -> str:
    """Create API token with specific scopes."""
    data = {
        "sub": user_id,
        "scopes": scopes,
        "type": TokenType.API_KEY.value
    }
    return create_access_token(data, expires_delta=timedelta(days=365))  # Long-lived API tokens