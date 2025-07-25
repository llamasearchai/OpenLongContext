# Authentication & Security Documentation

## Overview

The OpenLongContext API implements a comprehensive authentication and security system with the following features:

- **JWT Token-based Authentication**: Secure stateless authentication using JSON Web Tokens
- **API Key Authentication**: Long-lived API keys for programmatic access
- **User Management**: Complete user lifecycle management with registration, verification, and profile updates
- **Role-Based Access Control (RBAC)**: Fine-grained permissions and role management
- **Rate Limiting**: Configurable rate limits to prevent abuse
- **Security Headers**: Industry-standard security headers for protection
- **Session Management**: Secure session handling with refresh tokens
- **Account Security**: Password policies, account lockout, and secure password reset

## Authentication Methods

### 1. JWT Bearer Token

The primary authentication method for interactive users.

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"

# Use the token
curl -X GET http://localhost:8000/api/v1/docs/ \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### 2. API Key

For programmatic access and service-to-service communication.

```bash
# Create API key (requires authentication)
curl -X POST http://localhost:8000/auth/api-keys \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Production API Key", "expires_in_days": 365}'

# Use API key
curl -X GET http://localhost:8000/api/v1/docs/ \
  -H "X-API-Key: olc_your-api-key-here"
```

## User Registration & Management

### Registration

```python
import httpx

# Register new user
response = httpx.post(
    "http://localhost:8000/auth/register",
    json={
        "email": "user@example.com",
        "username": "newuser",
        "password": "SecurePass123!",
        "full_name": "New User"
    }
)

user = response.json()
```

### Email Verification

If email verification is enabled, users must verify their email before accessing protected resources:

```python
# Verify email with token
response = httpx.post(
    f"http://localhost:8000/auth/verify-email/{verification_token}"
)
```

### Password Reset

```python
# Request password reset
response = httpx.post(
    "http://localhost:8000/auth/forgot-password",
    json={"email": "user@example.com"}
)

# Reset password with token
response = httpx.post(
    "http://localhost:8000/auth/reset-password",
    json={
        "token": "reset-token",
        "new_password": "NewSecurePass123!"
    }
)
```

## Role-Based Access Control (RBAC)

### Default Roles

1. **admin**: Full system access
   - Permissions: `["*"]`
   
2. **user**: Standard user access
   - Permissions: `["read", "write", "delete_own"]`
   
3. **api_user**: API access for services
   - Permissions: `["read", "write", "api_access"]`
   
4. **readonly**: Read-only access
   - Permissions: `["read"]`

### Using Role-Based Protection

```python
from openlongcontext.api.auth import RequireRole, RequirePermission

# Require specific role
@router.get("/admin-only")
async def admin_endpoint(
    current_user: User = Depends(RequireRole(["admin"]))
):
    return {"message": "Admin access granted"}

# Require specific permission
@router.post("/write-operation")
async def write_endpoint(
    current_user: User = Depends(RequirePermission(["write"]))
):
    return {"message": "Write permission granted"}
```

## Security Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Authentication
SECRET_KEY=your-secret-key-here-change-in-production
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=30
AUTH_REFRESH_TOKEN_EXPIRE_DAYS=7

# Password Policy
AUTH_MIN_PASSWORD_LENGTH=8
AUTH_REQUIRE_UPPERCASE=true
AUTH_REQUIRE_LOWERCASE=true
AUTH_REQUIRE_DIGITS=true
AUTH_REQUIRE_SPECIAL_CHARS=true

# Rate Limiting
AUTH_RATE_LIMIT_ENABLED=true
AUTH_RATE_LIMIT_DEFAULT=100/minute
AUTH_RATE_LIMIT_AUTH=5/minute
AUTH_RATE_LIMIT_API=1000/hour

# Account Security
AUTH_MAX_LOGIN_ATTEMPTS=5
AUTH_LOCKOUT_DURATION_MINUTES=30
AUTH_EMAIL_VERIFICATION_REQUIRED=true
```

### Security Headers

The API automatically adds these security headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`

## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/register` | Register new user | No |
| POST | `/auth/login` | Login with credentials | No |
| POST | `/auth/refresh` | Refresh access token | Refresh Token |
| POST | `/auth/logout` | Logout current session | Yes |
| POST | `/auth/logout-all` | Logout all sessions | Yes |
| GET | `/auth/me` | Get current user profile | Yes |
| PUT | `/auth/me` | Update user profile | Yes |
| POST | `/auth/change-password` | Change password | Yes |
| POST | `/auth/verify-email/{token}` | Verify email address | No |
| POST | `/auth/resend-verification` | Resend verification email | Yes |
| POST | `/auth/forgot-password` | Request password reset | No |
| POST | `/auth/reset-password` | Reset password with token | No |

### API Key Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/api-keys` | Create new API key | Yes (Verified) |
| GET | `/auth/api-keys` | List user's API keys | Yes (Verified) |
| DELETE | `/auth/api-keys/{key_id}` | Delete API key | Yes (Verified) |

### Admin Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/auth/users` | List all users | Superuser |
| GET | `/auth/users/{user_id}` | Get user details | Superuser |
| DELETE | `/auth/users/{user_id}` | Delete user | Superuser |

## Rate Limiting

Rate limits are applied per IP address by default:

- **Authentication endpoints**: 5 requests/minute
- **API endpoints**: 1000 requests/hour
- **Default**: 100 requests/minute

Custom rate limits can be set for API keys:

```python
# Create API key with custom rate limit
response = httpx.post(
    "http://localhost:8000/auth/api-keys",
    json={
        "name": "High-volume API Key",
        "rate_limit": "10000/hour",
        "permissions": ["read", "write"]
    }
)
```

## Best Practices

### 1. Secure Token Storage

- **Access Tokens**: Store in memory or secure storage
- **Refresh Tokens**: Store as HTTP-only cookies or secure storage
- **API Keys**: Store securely and never expose in client-side code

### 2. Password Requirements

Passwords must meet these requirements by default:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character

### 3. Session Security

- Sessions expire after 30 minutes of inactivity
- Refresh tokens can extend sessions
- Maximum 5 concurrent sessions per user
- All sessions can be revoked on password change

### 4. API Key Security

- Use API keys only for server-to-server communication
- Set appropriate expiration dates
- Use IP whitelisting for additional security
- Rotate keys regularly

### 5. HTTPS in Production

Always use HTTPS in production:
- Set secure cookies
- Enable HSTS headers
- Use strong SSL/TLS configuration

## Error Handling

The API returns standard HTTP status codes:

- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `429 Too Many Requests`: Rate limit exceeded

Example error response:
```json
{
    "detail": "Not authenticated"
}
```

## Testing Authentication

Run the authentication tests:

```bash
pytest tests/unit/test_auth.py -v
```

## Production Checklist

Before deploying to production:

1. ✅ Change the `SECRET_KEY` to a strong random value
2. ✅ Use a production database (PostgreSQL/MySQL)
3. ✅ Enable HTTPS with valid SSL certificates
4. ✅ Configure email service for verification/reset emails
5. ✅ Set appropriate rate limits
6. ✅ Enable monitoring and logging
7. ✅ Regular security audits and updates
8. ✅ Implement backup and recovery procedures

## Integration Examples

### Python Client

```python
from openlongcontext.client import OpenLongContextClient

# Initialize client with API key
client = OpenLongContextClient(
    base_url="https://api.openlongcontext.ai",
    api_key="olc_your-api-key-here"
)

# Or with JWT token
client = OpenLongContextClient(
    base_url="https://api.openlongcontext.ai",
    token="your-jwt-token"
)

# Make authenticated requests
documents = client.list_documents()
```

### JavaScript/TypeScript

```typescript
// Using fetch with JWT
const response = await fetch('https://api.openlongcontext.ai/api/v1/docs/', {
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json'
  }
});

// Using fetch with API key
const response = await fetch('https://api.openlongcontext.ai/api/v1/docs/', {
  headers: {
    'X-API-Key': apiKey,
    'Content-Type': 'application/json'
  }
});
```

## Troubleshooting

### Common Issues

1. **"Not authenticated" error**
   - Ensure the token/API key is included in the request
   - Check if the token has expired
   - Verify the correct header format

2. **"Email not verified" error**
   - Complete email verification process
   - Request new verification email if needed

3. **"Account locked" error**
   - Wait for lockout period to expire
   - Contact admin to unlock account

4. **Rate limit exceeded**
   - Implement exponential backoff
   - Request higher rate limits if needed
   - Use API keys with custom limits

## Support

For security issues or questions:
- Email: security@openlongcontext.ai
- Documentation: https://docs.openlongcontext.ai
- GitHub Issues: https://github.com/openlongcontext/issues