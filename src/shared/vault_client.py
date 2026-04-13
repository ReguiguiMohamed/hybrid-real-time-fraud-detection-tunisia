"""
HashiCorp Vault Adapter for Amastan Fraud Shield Guard
Provides enterprise-grade secret management instead of .env files.

Usage:
    from src.shared.vault_client import get_secret, get_vault_client

    # Get a single secret
    api_token = get_secret("fraud-api", "api-token")

    # Get all secrets for a service
    secrets = get_vault_client().get_all_secrets("fraud-api")

In production, configure Vault with:
    VAULT_ADDR=https://vault.company.com
    VAULT_TOKEN=<app-role-token>
    Or use Kubernetes service account auth (approle)
"""
import os
import logging
import json
from typing import Optional
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_VAULT_ADDR = os.getenv("VAULT_ADDR", "http://localhost:8200")
DEFAULT_VAULT_TOKEN = os.getenv("VAULT_TOKEN", "")
DEFAULT_VAULT_PATH = os.getenv("VAULT_PATH", "secret/data/amastan")
VAULT_CACHE_TTL = 300  # 5 minutes


class VaultClient:
    """
    Enterprise secret management via HashiCorp Vault.
    Falls back to environment variables if Vault is unavailable.
    """

    _instance = None
    _cache = {}
    _cache_expiry = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.vault_addr = DEFAULT_VAULT_ADDR
            self.vault_token = DEFAULT_VAULT_TOKEN
            self.vault_path = DEFAULT_VAULT_PATH
            self._initialized = True
            self._hvac_client = None
            self._is_available = False

            # Test Vault connectivity
            self._test_connectivity()

    def _get_hvac_client(self):
        """Get or initialize the hvac Vault client."""
        if self._hvac_client is None:
            try:
                import hvac
                self._hvac_client = hvac.Client(
                    url=self.vault_addr,
                    token=self.vault_token,
                )
                self._is_available = self._hvac_client.is_authenticated()
                logger.info(f"Vault client initialized: {self.vault_addr} (authenticated: {self._is_available})")
            except ImportError:
                logger.warning("hvac package not installed. Falling back to environment variables.")
                self._hvac_client = None
                self._is_available = False
            except Exception as e:
                logger.warning(f"Vault connection failed: {e}. Falling back to environment variables.")
                self._hvac_client = None
                self._is_available = False

        return self._hvac_client

    def _test_connectivity(self):
        """Test if Vault is reachable and authenticated."""
        client = self._get_hvac_client()
        if client and self._is_available:
            try:
                client.read("sys/health")
                logger.info("Vault connectivity test passed")
            except Exception as e:
                logger.warning(f"Vault health check failed: {e}")
                self._is_available = False

    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get a secret from the local cache if not expired."""
        if key in self._cache:
            if datetime.now() < self._cache_expiry.get(key, datetime.now()):
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_expiry[key]
        return None

    def _set_cache(self, key: str, value: str):
        """Store a secret in the local cache with TTL."""
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + timedelta(seconds=VAULT_CACHE_TTL)

    def get_secret(self, secret_path: str, key: str, fallback_env: Optional[str] = None) -> Optional[str]:
        """
        Get a secret from Vault with environment variable fallback.

        Args:
            secret_path: Vault path (e.g., "secret/data/amastan/fraud-api")
            key: Secret key (e.g., "api-token")
            fallback_env: Environment variable name for fallback (e.g., "API_TOKEN")

        Returns:
            Secret value or None if not found.
        """
        cache_key = f"{secret_path}:{key}"

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Try Vault
        client = self._get_hvac_client()
        if client and self._is_available:
            try:
                response = client.secrets.kv.v2.read_secret_version(
                    path=secret_path,
                    mount_point="secret",
                )
                value = response["data"]["data"].get(key)
                if value is not None:
                    self._set_cache(cache_key, value)
                    return value
            except Exception as e:
                logger.warning(f"Vault read failed for {secret_path}:{key}: {e}")

        # Fallback to environment variable
        if fallback_env:
            value = os.getenv(fallback_env)
            if value:
                self._set_cache(cache_key, value)
                return value

        logger.error(f"Secret not found: {secret_path}:{key} (no Vault, no env var)")
        return None

    def get_all_secrets(self, secret_path: str) -> dict:
        """Get all secrets from a Vault path."""
        client = self._get_hvac_client()
        if client and self._is_available:
            try:
                response = client.secrets.kv.v2.read_secret_version(
                    path=secret_path,
                    mount_point="secret",
                )
                return response["data"]["data"]
            except Exception as e:
                logger.warning(f"Vault read all failed for {secret_path}: {e}")
        return {}

    def write_secret(self, secret_path: str, data: dict):
        """Write secrets to Vault."""
        client = self._get_hvac_client()
        if client and self._is_available:
            try:
                client.secrets.kv.v2.create_or_update_secret(
                    path=secret_path,
                    secret=data,
                    mount_point="secret",
                )
                logger.info(f"Secret written to {secret_path}")
            except Exception as e:
                logger.error(f"Vault write failed for {secret_path}: {e}")
                raise
        else:
            raise RuntimeError("Vault is not available")

    def is_available(self) -> bool:
        """Check if Vault is available."""
        return self._is_available


# Module-level convenience functions
_vault_client = None


def get_vault_client() -> VaultClient:
    """Get the Vault client singleton."""
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient()
    return _vault_client


def get_secret(key: str, fallback_env: Optional[str] = None) -> Optional[str]:
    """
    Get a secret from the default Amastan Vault path.

    Args:
        key: Secret key (e.g., "api-token")
        fallback_env: Environment variable for fallback

    Returns:
        Secret value.
    """
    client = get_vault_client()
    return client.get_secret(DEFAULT_VAULT_PATH, key, fallback_env)


def get_kafka_credentials() -> dict:
    """Get Kafka credentials from Vault."""
    return {
        "bootstrap_servers": get_secret("kafka-bootstrap", fallback_env="KAFKA_BOOTSTRAP_SERVERS") or "localhost:9092",
        "security_protocol": get_secret("kafka-security-protocol", fallback_env="KAFKA_SECURITY_PROTOCOL") or "PLAINTEXT",
        "sasl_mechanism": get_secret("kafka-sasl-mechanism", fallback_env="KAFKA_SASL_MECHANISM"),
        "sasl_username": get_secret("kafka-sasl-username", fallback_env="KAFKA_SASL_USERNAME"),
        "sasl_password": get_secret("kafka-sasl-password", fallback_env="KAFKA_SASL_PASSWORD"),
    }


def get_database_credentials() -> dict:
    """Get database credentials from Vault (if using external DB)."""
    return {
        "host": get_secret("db-host", fallback_env="DB_HOST") or "localhost",
        "port": get_secret("db-port", fallback_env="DB_PORT") or "5432",
        "username": get_secret("db-username", fallback_env="DB_USERNAME"),
        "password": get_secret("db-password", fallback_env="DB_PASSWORD"),
        "database": get_secret("db-name", fallback_env="DB_NAME") or "amastan",
    }
