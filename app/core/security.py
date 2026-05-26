"""
Authentication — JWT bearer tokens + optional API key header.
For the Canadian job market: show you know auth patterns, not just ML.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

import structlog
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel

from app.core.config import settings

log = structlog.get_logger()

bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


# ── Token models ───────────────────────────────────────────────────────────

class TokenData(BaseModel):
    sub: str  # subject (user id or service name)
    scopes: list[str] = []
    exp: Optional[datetime] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ── Token creation (for /auth/token endpoint) ──────────────────────────────

def create_access_token(subject: str, scopes: list[str] = []) -> Token:
    expiry = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRY_MINUTES)
    payload = {
        "sub": subject,
        "scopes": scopes,
        "exp": expiry,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return Token(
        access_token=token,
        expires_in=settings.JWT_EXPIRY_MINUTES * 60,
    )


# ── Token verification ─────────────────────────────────────────────────────

def _decode_jwt(token: str) -> TokenData:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenData(**payload)
    except JWTError as e:
        log.warning("jwt.invalid", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ───────────────────────────────────────────────────

async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
    api_key: Annotated[Optional[str], Security(api_key_header)],
) -> TokenData:
    """
    Accept either:
      - Bearer JWT: Authorization: Bearer <token>
      - API key:    X-API-Key: <key>

    This dual approach lets you issue long-lived API keys for
    service-to-service calls and short-lived JWTs for user sessions.
    """
    if credentials:
        return _decode_jwt(credentials.credentials)

    if api_key:
        # In production, validate against a DB or secrets manager
        # Here we validate against a hashed env var for simplicity
        import hashlib
        expected = getattr(settings, "HASHED_API_KEY", None)
        if expected and hashlib.sha256(api_key.encode()).hexdigest() == expected:
            return TokenData(sub="api_key_user", scopes=["predict"])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required: provide Bearer token or X-API-Key header",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Convenience dependency aliases
CurrentUser = Annotated[TokenData, Depends(get_current_user)]
