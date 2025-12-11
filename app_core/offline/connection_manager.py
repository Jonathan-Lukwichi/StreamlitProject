# =============================================================================
# app_core/offline/connection_manager.py
# Connection Status Detection and Management
# =============================================================================
"""
ConnectionManager - Detects and monitors internet/Supabase connectivity.

Features:
- Automatic connection detection
- Periodic health checks
- Event callbacks for status changes
- Thread-safe singleton pattern
"""

from __future__ import annotations
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status states."""
    ONLINE = "online"           # Full connectivity (Internet + Supabase)
    OFFLINE = "offline"         # No connectivity
    DEGRADED = "degraded"       # Internet OK but Supabase unavailable
    CHECKING = "checking"       # Currently checking status
    UNKNOWN = "unknown"         # Initial state


@dataclass
class ConnectionState:
    """Current connection state with metadata."""
    status: ConnectionStatus = ConnectionStatus.UNKNOWN
    internet_available: bool = False
    supabase_available: bool = False
    last_check: Optional[datetime] = None
    last_online: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None


class ConnectionManager:
    """
    Singleton manager for connection status detection.

    Usage:
        manager = ConnectionManager.get_instance()
        if manager.is_online:
            # Use cloud services
        else:
            # Use local fallback
    """

    _instance: Optional[ConnectionManager] = None
    _lock = threading.Lock()

    # Configuration
    CHECK_INTERVAL_ONLINE = 30      # Seconds between checks when online
    CHECK_INTERVAL_OFFLINE = 10     # Seconds between checks when offline
    CONNECTION_TIMEOUT = 5          # Timeout for connection tests
    SUPABASE_HEALTH_ENDPOINT = None # Set dynamically from env

    def __init__(self):
        """Initialize connection manager (use get_instance() instead)."""
        self._state = ConnectionState()
        self._callbacks: List[Callable[[ConnectionState], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> ConnectionManager:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ConnectionManager()
        return cls._instance

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def status(self) -> ConnectionStatus:
        """Get current status."""
        return self._state.status

    @property
    def is_online(self) -> bool:
        """Check if we have full connectivity."""
        return self._state.status == ConnectionStatus.ONLINE

    @property
    def is_offline(self) -> bool:
        """Check if we're completely offline."""
        return self._state.status == ConnectionStatus.OFFLINE

    @property
    def has_internet(self) -> bool:
        """Check if internet is available (even if Supabase isn't)."""
        return self._state.internet_available

    @property
    def has_supabase(self) -> bool:
        """Check if Supabase is available."""
        return self._state.supabase_available

    def initialize(self, start_monitoring: bool = True) -> None:
        """
        Initialize the connection manager.

        Args:
            start_monitoring: Whether to start background monitoring
        """
        if self._initialized:
            return

        # Initial check
        self.check_connection()

        # Start background monitoring
        if start_monitoring:
            self.start_monitoring()

        self._initialized = True
        logger.info(f"ConnectionManager initialized. Status: {self._state.status.value}")

    def check_connection(self) -> ConnectionState:
        """
        Perform a connection check and update state.

        Returns:
            Updated ConnectionState
        """
        old_status = self._state.status
        self._state.status = ConnectionStatus.CHECKING
        self._state.last_check = datetime.now()

        # Check internet connectivity
        internet_ok = self._check_internet()
        self._state.internet_available = internet_ok

        # Check Supabase if internet is available
        supabase_ok = False
        if internet_ok:
            supabase_ok = self._check_supabase()
        self._state.supabase_available = supabase_ok

        # Determine overall status
        if internet_ok and supabase_ok:
            self._state.status = ConnectionStatus.ONLINE
            self._state.last_online = datetime.now()
            self._state.consecutive_failures = 0
            self._state.error_message = None
        elif internet_ok and not supabase_ok:
            self._state.status = ConnectionStatus.DEGRADED
            self._state.consecutive_failures += 1
        else:
            self._state.status = ConnectionStatus.OFFLINE
            self._state.consecutive_failures += 1

        # Notify callbacks if status changed
        if old_status != self._state.status:
            self._notify_callbacks()
            logger.info(f"Connection status changed: {old_status.value} -> {self._state.status.value}")

        return self._state

    def _check_internet(self) -> bool:
        """
        Check internet connectivity by attempting to reach well-known hosts.

        Returns:
            True if internet is available
        """
        hosts = [
            ("8.8.8.8", 53),      # Google DNS
            ("1.1.1.1", 53),      # Cloudflare DNS
            ("208.67.222.222", 53), # OpenDNS
        ]

        for host, port in hosts:
            try:
                socket.setdefaulttimeout(self.CONNECTION_TIMEOUT)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    return True
            except (socket.error, socket.timeout, OSError):
                continue

        return False

    def _check_supabase(self) -> bool:
        """
        Check Supabase connectivity.

        Returns:
            True if Supabase is reachable
        """
        try:
            # Try to import and check Supabase
            supabase_url = os.getenv("SUPABASE_URL", "")

            if not supabase_url:
                # No Supabase configured - treat as available (local-only mode)
                return True

            # Extract host from URL and try to connect
            from urllib.parse import urlparse
            parsed = urlparse(supabase_url)
            host = parsed.hostname
            port = parsed.port or 443

            if host:
                socket.setdefaulttimeout(self.CONNECTION_TIMEOUT)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0

        except Exception as e:
            self._state.error_message = str(e)
            logger.debug(f"Supabase check failed: {e}")

        return False

    def start_monitoring(self) -> None:
        """Start background connection monitoring."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ConnectionMonitor"
        )
        self._monitor_thread.start()
        logger.debug("Connection monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background connection monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.debug("Connection monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            # Determine check interval based on current status
            interval = (
                self.CHECK_INTERVAL_ONLINE
                if self.is_online
                else self.CHECK_INTERVAL_OFFLINE
            )

            # Wait for interval or stop signal
            if self._stop_monitoring.wait(timeout=interval):
                break

            # Perform check
            try:
                self.check_connection()
            except Exception as e:
                logger.error(f"Error in connection check: {e}")

    def register_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """
        Register a callback for connection status changes.

        Args:
            callback: Function called with ConnectionState when status changes
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[ConnectionState], None]) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of status change."""
        for callback in self._callbacks:
            try:
                callback(self._state)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    def force_offline(self) -> None:
        """Force offline mode (for testing or user preference)."""
        self._state.status = ConnectionStatus.OFFLINE
        self._state.internet_available = False
        self._state.supabase_available = False
        self._notify_callbacks()
        logger.info("Forced offline mode")

    def force_check(self) -> ConnectionState:
        """Force an immediate connection check."""
        return self.check_connection()

    def get_status_display(self) -> dict:
        """Get status information for UI display."""
        return {
            "status": self._state.status.value,
            "is_online": self.is_online,
            "internet": self._state.internet_available,
            "supabase": self._state.supabase_available,
            "last_check": self._state.last_check.isoformat() if self._state.last_check else None,
            "last_online": self._state.last_online.isoformat() if self._state.last_online else None,
            "failures": self._state.consecutive_failures,
            "error": self._state.error_message,
        }


# Singleton accessor
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """
    Get the global ConnectionManager instance.

    Returns:
        ConnectionManager singleton
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager.get_instance()
        _connection_manager.initialize()
    return _connection_manager


# Convenience functions
def is_online() -> bool:
    """Quick check if we're online."""
    return get_connection_manager().is_online


def is_offline() -> bool:
    """Quick check if we're offline."""
    return get_connection_manager().is_offline
