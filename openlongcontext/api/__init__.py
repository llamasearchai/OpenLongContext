from fastapi import FastAPI
from contextlib import asynccontextmanager
from . import routes
from .auth import auth_router
from .middleware import setup_middleware
from .database import init_db, create_initial_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Initializing database...")
    init_db()
    create_initial_data()
    print("OpenLongContext API started successfully!")
    
    yield
    
    # Shutdown
    print("Shutting down OpenLongContext API...")


app = FastAPI(
    title="OpenLongContext Document QA API",
    description="Production-ready API for long-context document upload, retrieval, and question answering using efficient transformer models.",
    version="1.0.0",
    contact={
        "name": "Nik Jois",
        "email": "nikjois@llamasearch.ai",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup middleware
setup_middleware(app)

# Include routers
app.include_router(auth_router)
app.include_router(routes.router) 