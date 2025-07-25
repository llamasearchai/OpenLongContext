"""Database configuration and session management."""
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./openlongcontext.db"
)

# Handle PostgreSQL URL format for production
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
else:
    # PostgreSQL/MySQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from openlongcontext.api.auth.models import Base
    Base.metadata.create_all(bind=engine)


def create_initial_data():
    """Create initial data like default roles and superuser."""
    from openlongcontext.api.auth.models import Role, User

    db = SessionLocal()
    try:
        # Create default roles if they don't exist
        default_roles = [
            {
                "name": "admin",
                "description": "Administrator with full access",
                "permissions": ["*"]
            },
            {
                "name": "user",
                "description": "Regular user",
                "permissions": ["read", "write", "delete_own"]
            },
            {
                "name": "api_user",
                "description": "API user with programmatic access",
                "permissions": ["read", "write", "api_access"]
            },
            {
                "name": "readonly",
                "description": "Read-only access",
                "permissions": ["read"]
            }
        ]

        for role_data in default_roles:
            role = db.query(Role).filter(Role.name == role_data["name"]).first()
            if not role:
                role = Role(**role_data)
                db.add(role)

        db.commit()

        # Create superuser if it doesn't exist
        superuser_email = os.getenv("SUPERUSER_EMAIL", "admin@openlongcontext.ai")
        superuser_password = os.getenv("SUPERUSER_PASSWORD", "changeme123!")

        existing_superuser = db.query(User).filter(User.email == superuser_email).first()
        if not existing_superuser:
            from openlongcontext.api.auth.services import UserService
            user_service = UserService(db)

            superuser = user_service.create_user(
                email=superuser_email,
                username="admin",
                password=superuser_password,
                full_name="System Administrator",
                is_superuser=True,
                is_verified=True
            )

            # Add admin role to superuser
            admin_role = db.query(Role).filter(Role.name == "admin").first()
            if admin_role:
                superuser.roles.append(admin_role)
                db.commit()

            print(f"Created superuser: {superuser_email}")
            print("Please change the default password immediately!")

    finally:
        db.close()
