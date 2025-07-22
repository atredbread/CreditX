"""
JSON Cache Manager

This module provides a centralized way to manage JSON cache files for agent data.
It handles file creation, validation, and expiration.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
CACHE_FILE = CACHE_DIR / 'agent_data_cache.json'
CACHE_EXPIRY_DAYS = 1


def ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_cache_valid() -> bool:
    """
    Check if the cache file exists and is not expired.
    
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not CACHE_FILE.exists():
        return False
    
    # Check file modification time
    file_mod_time = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
    expiry_time = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
    
    return file_mod_time > expiry_time


def load_cache() -> Dict[str, Any]:
    """
    Load data from the cache file.
    
    Returns:
        Dict containing the cached data or empty dict if cache is invalid
    """
    if not is_cache_valid():
        logger.info("Cache is invalid or expired")
        return {}
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading cache: {e}")
        return {}


def save_cache(data: Dict[str, Any]) -> None:
    """
    Save data to the cache file.
    
    Args:
        data: Dictionary containing data to cache
    """
    try:
        ensure_cache_dir()
        temp_file = f"{CACHE_FILE}.tmp"
        
        # Write to temporary file first
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        if os.path.exists(temp_file):
            if CACHE_FILE.exists():
                os.remove(CACHE_FILE)
            os.rename(temp_file, CACHE_FILE)
            logger.info(f"Cache updated: {CACHE_FILE}")
            
    except (IOError, OSError) as e:
        logger.error(f"Error saving cache: {e}")
        # Clean up temporary file if it exists
        if os.path.exists(temp_file):
            os.remove(temp_file)


def clear_cache() -> None:
    """Remove the cache file if it exists."""
    if CACHE_FILE.exists():
        try:
            os.remove(CACHE_FILE)
            logger.info(f"Cache cleared: {CACHE_FILE}")
        except OSError as e:
            logger.error(f"Error clearing cache: {e}")


def update_agent_data(agent_id: str, agent_data: Dict[str, Any]) -> None:
    """
    Update cache with data for a specific agent.
    
    Args:
        agent_id: Unique identifier for the agent
        agent_data: Dictionary containing agent data to cache
    """
    cache = load_cache()
    if not cache:
        cache = {"agents": {}, "metadata": {"last_updated": datetime.now().isoformat()}}
    
    cache["agents"][agent_id] = agent_data
    cache["metadata"]["last_updated"] = datetime.now().isoformat()
    
    save_cache(cache)


def get_agent_data(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached data for a specific agent.
    
    Args:
        agent_id: Unique identifier for the agent
        
    Returns:
        Dictionary containing cached agent data or None if not found
    """
    cache = load_cache()
    return cache.get("agents", {}).get(agent_id)


def get_all_agents_data() -> Dict[str, Any]:
    """
    Get all cached agent data.
    
    Returns:
        Dictionary containing all cached agent data
    """
    return load_cache().get("agents", {})
