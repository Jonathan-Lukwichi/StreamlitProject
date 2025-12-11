# =============================================================================
# app_core/offline/__init__.py
# Offline-First Architecture for HealthForecast AI
# =============================================================================
"""
Offline-First Architecture Module

This module provides seamless offline functionality for the HealthForecast AI
platform. The app works identically whether internet is available or not.

Architecture:
------------
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE-FIRST ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                 UnifiedDataService                        │  │
│   │         (Single API - Apps use this only)                 │  │
│   └──────────────────────────────────────────────────────────┘  │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│   ┌──────────────────┐        ┌──────────────────┐             │
│   │  ConnectionMgr   │        │   CacheManager   │             │
│   │  (Online/Offline)│        │  (Files/Models)  │             │
│   └──────────────────┘        └──────────────────┘             │
│              │                                                   │
│   ┌──────────┴──────────┐                                       │
│   ▼                     ▼                                       │
│ ┌────────┐        ┌──────────┐                                  │
│ │Supabase│◄──────►│  SQLite  │                                  │
│ │(Cloud) │  Sync  │ (Local)  │                                  │
│ └────────┘        └──────────┘                                  │
│              ▲                                                   │
│              │                                                   │
│   ┌──────────────────┐                                          │
│   │   SyncEngine     │                                          │
│   │ (Auto Background)│                                          │
│   └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘

Usage:
------
from app_core.offline import UnifiedDataService, get_data_service

# Get the singleton service
service = get_data_service()

# Use it - automatically handles online/offline
df = service.fetch_patient_data()
service.save_patient_data(df)

# Check status
print(service.is_online)  # True/False
print(service.pending_sync_count)  # Number of pending operations
"""

from app_core.offline.connection_manager import (
    ConnectionManager,
    get_connection_manager,
    ConnectionStatus,
)

from app_core.offline.local_database import (
    LocalDatabase,
    get_local_database,
)

from app_core.offline.cache_manager import (
    CacheManager,
    get_cache_manager,
)

from app_core.offline.sync_engine import (
    SyncEngine,
    get_sync_engine,
    SyncOperation,
    SyncStatus,
)

from app_core.offline.unified_data_service import (
    UnifiedDataService,
    get_data_service,
)

__all__ = [
    # Connection Management
    "ConnectionManager",
    "get_connection_manager",
    "ConnectionStatus",
    # Local Database
    "LocalDatabase",
    "get_local_database",
    # Cache Management
    "CacheManager",
    "get_cache_manager",
    # Sync Engine
    "SyncEngine",
    "get_sync_engine",
    "SyncOperation",
    "SyncStatus",
    # Unified Service (Main API)
    "UnifiedDataService",
    "get_data_service",
]
