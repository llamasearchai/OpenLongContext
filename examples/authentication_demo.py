"""
Authentication Demo for OpenLongContext API

This script demonstrates how to use the authentication system:
1. Register a new user
2. Login and get access token
3. Create an API key
4. Make authenticated requests
5. Manage user profile
"""

import httpx
import asyncio
from typing import Optional


class OpenLongContextAuth:
    """Simple authentication client for OpenLongContext API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.api_key: Optional[str] = None
    
    async def register(self, email: str, username: str, password: str, full_name: str = None):
        """Register a new user."""
        response = await self.client.post(
            "/auth/register",
            json={
                "email": email,
                "username": username,
                "password": password,
                "full_name": full_name
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def login(self, username: str, password: str):
        """Login and store tokens."""
        response = await self.client.post(
            "/auth/login",
            data={
                "username": username,
                "password": password
            }
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        
        # Update client headers
        self.client.headers["Authorization"] = f"Bearer {self.access_token}"
        
        return data
    
    async def create_api_key(self, name: str, expires_in_days: int = None):
        """Create an API key."""
        if not self.access_token:
            raise Exception("Must be logged in to create API key")
        
        response = await self.client.post(
            "/auth/api-keys",
            json={
                "name": name,
                "expires_in_days": expires_in_days
            }
        )
        response.raise_for_status()
        
        data = response.json()
        self.api_key = data["key"]
        return data
    
    async def get_profile(self):
        """Get current user profile."""
        if not self.access_token:
            raise Exception("Must be logged in")
        
        response = await self.client.get("/auth/me")
        response.raise_for_status()
        return response.json()
    
    async def update_profile(self, **kwargs):
        """Update user profile."""
        if not self.access_token:
            raise Exception("Must be logged in")
        
        response = await self.client.put("/auth/me", json=kwargs)
        response.raise_for_status()
        return response.json()
    
    async def list_documents(self, use_api_key: bool = False):
        """List documents using either JWT or API key."""
        headers = {}
        if use_api_key and self.api_key:
            headers["X-API-Key"] = self.api_key
            # Remove Authorization header for API key test
            headers["Authorization"] = ""
        
        response = await self.client.get(
            "/api/v1/docs/",
            headers=headers if headers else None
        )
        response.raise_for_status()
        return response.json()
    
    async def upload_document(self, file_path: str):
        """Upload a document."""
        if not self.access_token and not self.api_key:
            raise Exception("Must be authenticated")
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "text/plain")}
            response = await self.client.post(
                "/api/v1/docs/upload",
                files=files
            )
        
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


async def main():
    """Run authentication demo."""
    print("🔐 OpenLongContext Authentication Demo\n")
    
    # Initialize client
    auth = OpenLongContextAuth()
    
    try:
        # 1. Register a new user
        print("1. Registering new user...")
        try:
            user = await auth.register(
                email="demo@example.com",
                username="demouser",
                password="DemoPass123!",
                full_name="Demo User"
            )
            print(f"[SUCCESS] User registered: {user['username']} ({user['email']})")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                print("[WARNING] User already exists, continuing...")
            else:
                raise
        
        # 2. Login
        print("\n2. Logging in...")
        tokens = await auth.login("demouser", "DemoPass123!")
        print(f"[SUCCESS] Logged in successfully!")
        print(f"   Access token expires in: {tokens['expires_in']} seconds")
        
        # 3. Get user profile
        print("\n3. Getting user profile...")
        profile = await auth.get_profile()
        print(f"[SUCCESS] Profile retrieved:")
        print(f"   Username: {profile['username']}")
        print(f"   Email: {profile['email']}")
        print(f"   Full Name: {profile['full_name']}")
        print(f"   Verified: {profile['is_verified']}")
        
        # 4. Update profile
        print("\n4. Updating profile...")
        updated_profile = await auth.update_profile(
            full_name="Demo User Updated"
        )
        print(f"[SUCCESS] Profile updated: {updated_profile['full_name']}")
        
        # 5. Create API key
        print("\n5. Creating API key...")
        api_key = await auth.create_api_key(
            name="Demo API Key",
            expires_in_days=30
        )
        print(f"[SUCCESS] API key created:")
        print(f"   ID: {api_key['id']}")
        print(f"   Key: {api_key['key']}")
        print(f"   [WARNING] Save this key securely - it won't be shown again!")
        
        # 6. Test authenticated endpoints
        print("\n6. Testing authenticated endpoints...")
        
        # Test with JWT
        print("   Testing with JWT token...")
        docs_jwt = await auth.list_documents(use_api_key=False)
        print(f"   [SUCCESS] JWT auth successful - Found {len(docs_jwt)} documents")
        
        # Test with API key
        print("   Testing with API key...")
        docs_api = await auth.list_documents(use_api_key=True)
        print(f"   [SUCCESS] API key auth successful - Found {len(docs_api)} documents")
        
        # 7. Test file upload (if you have a test file)
        print("\n7. Testing file upload...")
        try:
            # Create a test file
            test_file = "test_document.txt"
            with open(test_file, "w") as f:
                f.write("This is a test document for OpenLongContext authentication demo.")
            
            upload_result = await auth.upload_document(test_file)
            print(f"[SUCCESS] Document uploaded:")
            print(f"   Document ID: {upload_result['doc_id']}")
            print(f"   Message: {upload_result['message']}")
            
            # Clean up
            import os
            os.remove(test_file)
        except Exception as e:
            print(f"[WARNING] Upload test skipped: {e}")
        
        print("\n[SUCCESS] Authentication demo completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
    finally:
        await auth.close()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())