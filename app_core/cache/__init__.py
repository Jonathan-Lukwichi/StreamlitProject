# app_core/cache/__init__.py
"""
Cache module for persisting session data to disk.
Avoids re-running data processing and model training on app restart.
"""
from .cache_manager import (
    CacheManager,
    get_cache_manager,
    save_to_cache,
    load_from_cache,
    clear_cache,
    get_cache_info,
    has_cached_data,
)

__all__ = [
    "CacheManager",
    "get_cache_manager",
    "save_to_cache",
    "load_from_cache",
    "clear_cache",
    "get_cache_info",
    "has_cached_data",
]
