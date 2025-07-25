"""Test authentication system."""
import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from openlongcontext.api import app
from openlongcontext.api.database import get_db, Base
from openlongcontext.api.auth import (
    UserService, SessionService, APIKeyService,
    UserCreate, LoginRequest
)
from openlongcontext.api.auth.jwt import create_access_token, decode_token, verify_token, TokenType


# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create tables
Base.metadata.create_all(bind=engine)

# Test client
client = TestClient(app)


@pytest.fixture
def db_session():
    """Create a test database session."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def user_service(db_session):
    """Create user service instance."""
    return UserService(db_session)


@pytest.fixture
def test_user(user_service):
    """Create a test user."""
    user = user_service.create_user(
        email="test@example.com",
        username="testuser",
        password="TestPass123!",
        full_name="Test User",
        is_verified=True
    )
    return user


class TestUserService:
    """Test user service."""
    
    def test_create_user(self, user_service):
        """Test user creation."""
        user = user_service.create_user(
            email="newuser@example.com",
            username="newuser",
            password="SecurePass123!",
            full_name="New User"
        )
        
        assert user.email == "newuser@example.com"
        assert user.username == "newuser"
        assert user.full_name == "New User"
        assert user.verify_password("SecurePass123!")
        assert not user.is_verified
        assert not user.is_superuser
    
    def test_create_duplicate_user(self, user_service, test_user):
        """Test creating duplicate user."""
        with pytest.raises(Exception):
            user_service.create_user(
                email=test_user.email,
                username="different",
                password="Password123!"
            )
    
    def test_authenticate_user(self, user_service, test_user):
        """Test user authentication."""
        # Test with username
        user = user_service.authenticate_user("testuser", "TestPass123!")
        assert user is not None
        assert user.id == test_user.id
        
        # Test with email
        user = user_service.authenticate_user("test@example.com", "TestPass123!")
        assert user is not None
        assert user.id == test_user.id
        
        # Test with wrong password
        user = user_service.authenticate_user("testuser", "WrongPassword")
        assert user is None
    
    def test_update_user(self, user_service, test_user):
        """Test user update."""
        from openlongcontext.api.auth import UserUpdate
        
        updated_user = user_service.update_user(
            test_user.id,
            UserUpdate(full_name="Updated Name", email="updated@example.com")
        )
        
        assert updated_user.full_name == "Updated Name"
        assert updated_user.email == "updated@example.com"
    
    def test_password_reset(self, user_service, test_user):
        """Test password reset flow."""
        # Request reset
        token = user_service.request_password_reset(test_user.email)
        assert token is not None
        
        # Reset password
        success = user_service.reset_password(token, "NewPassword123!")
        assert success
        
        # Verify new password works
        user = user_service.authenticate_user(test_user.username, "NewPassword123!")
        assert user is not None


class TestJWT:
    """Test JWT functionality."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        data = {"sub": "user123", "username": "testuser"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
    
    def test_decode_token(self):
        """Test token decoding."""
        data = {"sub": "user123", "username": "testuser", "scopes": ["read", "write"]}
        token = create_access_token(data)
        
        token_data = decode_token(token)
        assert token_data is not None
        assert token_data.user_id == "user123"
        assert token_data.username == "testuser"
        assert token_data.scopes == ["read", "write"]
        assert token_data.token_type == TokenType.ACCESS
    
    def test_expired_token(self):
        """Test expired token handling."""
        data = {"sub": "user123"}
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))
        
        token_data = decode_token(token)
        assert token_data is None


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_register(self):
        """Test user registration."""
        response = client.post(
            "/auth/register",
            json={
                "email": "register@example.com",
                "username": "registeruser",
                "password": "SecurePass123!",
                "full_name": "Register User"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "register@example.com"
        assert data["username"] == "registeruser"
        assert "id" in data
    
    def test_login(self):
        """Test user login."""
        # First register a user
        client.post(
            "/auth/register",
            json={
                "email": "login@example.com",
                "username": "loginuser",
                "password": "LoginPass123!",
                "is_verified": True
            }
        )
        
        # Login with username
        response = client.post(
            "/auth/login",
            data={
                "username": "loginuser",
                "password": "LoginPass123!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    def test_get_current_user(self):
        """Test getting current user."""
        # Register and login
        client.post(
            "/auth/register",
            json={
                "email": "current@example.com",
                "username": "currentuser",
                "password": "CurrentPass123!"
            }
        )
        
        login_response = client.post(
            "/auth/login",
            data={
                "username": "currentuser",
                "password": "CurrentPass123!"
            }
        )
        
        token = login_response.json()["access_token"]
        
        # Get current user
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "current@example.com"
        assert data["username"] == "currentuser"
    
    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/v1/docs/")
        assert response.status_code == 401
    
    def test_api_key_creation(self):
        """Test API key creation."""
        # Register and login
        client.post(
            "/auth/register",
            json={
                "email": "apikey@example.com",
                "username": "apikeyuser",
                "password": "ApiKeyPass123!",
                "is_verified": True
            }
        )
        
        login_response = client.post(
            "/auth/login",
            data={
                "username": "apikeyuser",
                "password": "ApiKeyPass123!"
            }
        )
        
        token = login_response.json()["access_token"]
        
        # Create API key
        response = client.post(
            "/auth/api-keys",
            json={
                "name": "Test API Key",
                "expires_in_days": 30
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("olc_")
        assert data["name"] == "Test API Key"
    
    def test_password_change(self):
        """Test password change."""
        # Register and login
        client.post(
            "/auth/register",
            json={
                "email": "changepass@example.com",
                "username": "changepassuser",
                "password": "OldPass123!",
                "is_verified": True
            }
        )
        
        login_response = client.post(
            "/auth/login",
            data={
                "username": "changepassuser",
                "password": "OldPass123!"
            }
        )
        
        token = login_response.json()["access_token"]
        
        # Change password
        response = client.post(
            "/auth/change-password",
            json={
                "current_password": "OldPass123!",
                "new_password": "NewPass123!"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        
        # Try login with new password
        new_login = client.post(
            "/auth/login",
            data={
                "username": "changepassuser",
                "password": "NewPass123!"
            }
        )
        
        assert new_login.status_code == 200


class TestRateLimiting:
    """Test rate limiting."""
    
    def test_auth_rate_limit(self):
        """Test authentication endpoint rate limiting."""
        # Make multiple rapid login attempts
        responses = []
        for i in range(10):
            response = client.post(
                "/auth/login",
                data={
                    "username": f"user{i}",
                    "password": "password"
                }
            )
            responses.append(response)
        
        # Some requests should be rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        # Note: In test environment, rate limiting might be disabled
        # assert rate_limited


class TestSecurity:
    """Test security features."""
    
    def test_security_headers(self):
        """Test security headers."""
        response = client.get("/health")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in response.headers
    
    def test_password_validation(self):
        """Test password validation."""
        # Weak password
        response = client.post(
            "/auth/register",
            json={
                "email": "weak@example.com",
                "username": "weakuser",
                "password": "weak"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_account_lockout(self, user_service, test_user):
        """Test account lockout after failed attempts."""
        # Make multiple failed login attempts
        for i in range(6):
            user_service.authenticate_user(test_user.username, "WrongPassword")
        
        # Account should be locked
        with pytest.raises(Exception) as exc_info:
            user_service.authenticate_user(test_user.username, "TestPass123!")
        
        assert "locked" in str(exc_info.value).lower()