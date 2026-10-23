import os
from dotenv import load_dotenv

# Load .env from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").strip().lower()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")

# JWT Security Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
INSECURE_SECRETS = {"dev-secret", "test-secret", "secret", "change-me", "insecure"}

if ENVIRONMENT == "production":
    if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32 or JWT_SECRET_KEY in INSECURE_SECRETS:
        raise ValueError(
            "CRITICAL SECURITY CONFIGURATION ERROR: In production, JWT_SECRET_KEY "
            "must be explicitly set in the environment to a secure string of at least 32 characters."
        )
else:
    # Safe default for local development and offline test execution
    if not JWT_SECRET_KEY:
        JWT_SECRET_KEY = "adaptive-lsrw-gd-dev-secret-key-32chars-min-security"

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

# One-Time Password (OTP) Configuration
OTP_EXPIRY_SECONDS = int(os.getenv("OTP_EXPIRY_SECONDS", "300"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "3"))
OTP_COOLDOWN_SECONDS = int(os.getenv("OTP_COOLDOWN_SECONDS", "60"))

# Trusted Coordinator Whitelist (lowercased emails)
raw_coordinators = os.getenv("TRUSTED_COORDINATOR_EMAILS", "coordinator@example.com,admin@adaptive-gd.edu")
TRUSTED_COORDINATOR_EMAILS = frozenset(
    e.strip().lower() for e in raw_coordinators.split(",") if e.strip()
)

# LiveKit SFU Configuration (Optional until M9, but endpoints must fail gracefully if not configured)
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")