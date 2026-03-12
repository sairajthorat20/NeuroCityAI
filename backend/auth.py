"""
auth.py
-------
Authentication router for NeuroCity.
Endpoints:
    POST /auth/signup  — create account
    POST /auth/login   — login and receive JWT
    GET  /auth/me      — validate token and return profile
"""

import os
import bcrypt
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt

from database import get_user_by_email, create_user

router = APIRouter(prefix="/auth", tags=["Auth"])
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET: str = os.environ.get("JWT_SECRET", "neurocity-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 7  # tokens valid for 1 week

BCRYPT_ROUNDS = 12  # work factor for password hashing


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(user_id: int, email: str, name: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest):
    """Register a new user account."""
    if get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed = hash_password(body.password)
    create_user(body.full_name, body.email, hashed)
    user = get_user_by_email(body.email)
    token = create_token(user["id"], user["email"], user["full_name"])
    return {
        "token": token,
        "user": {"id": user["id"], "email": user["email"], "name": user["full_name"]},
    }


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    """Login with email + password, receive JWT."""
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_token(user["id"], user["email"], user["full_name"])
    return {
        "token": token,
        "user": {"id": user["id"], "email": user["email"], "name": user["full_name"]},
    }


@router.get("/me")
def get_me(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Return the current user profile from the JWT."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        payload = decode_token(credentials.credentials)
        return {"id": payload["sub"], "email": payload["email"], "name": payload["name"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
