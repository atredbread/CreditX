"""
Agent Data Store Module

This module provides a unified interface for storing, updating, and retrieving
agent data across the application. It serves as the single source of truth for
agent-related information and handles serialization to/from JSON format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class AgentDataStore:
    """
    Unified storage for agent data with JSON serialization support.
    
    This class provides methods to:
    - Store and retrieve agent data in a structured format
    - Merge data from different sources (repayments, sales, DPD, etc.)
    - Save/load data to/from JSON files
    - Query agent data efficiently
    """
    
    def __init__(self, data: Optional[Dict[str, Dict]] = None):
        """
        Initialize the AgentDataStore.
        
        Args:
            data: Optional initial data dictionary in the format {bzid: agent_data}
        """
        self.data = data or {}
        self.metadata = {
            'version': '1.0.0',
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'source_files': []
        }
    
    def add_agent_data(self, bzid: str, source: str, data: Dict) -> None:
        """
        Add or update agent data from a specific source.
        
        Args:
            bzid: Agent's unique identifier
            source: Data source identifier (e.g., 'repayments', 'sales', 'dpd')
            data: Dictionary containing the agent's data from this source
        """
        bzid = str(bzid)  # Ensure bzid is a string
        
        if bzid not in self.data:
            self.data[bzid] = {
                'bzid': bzid,
                'sources': {source: data},
                'metadata': {
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'sources': [source]
                }
            }
        else:
            # Update existing agent data
            agent = self.data[bzid]
            agent['sources'][source] = data
            agent['metadata']['updated_at'] = datetime.now(timezone.utc).isoformat()
            if source not in agent['metadata']['sources']:
                agent['metadata']['sources'].append(source)
    
    def get_agent(self, bzid: str) -> Optional[Dict]:
        """
        Retrieve all data for a specific agent.
        
        Args:
            bzid: Agent's unique identifier
            
        Returns:
            Dictionary containing the agent's data or None if not found
        """
        return self.data.get(str(bzid))
    
    def get_agents(self, bzids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Retrieve data for multiple agents.
        
        Args:
            bzids: List of agent IDs to retrieve. If None, returns all agents.
            
        Returns:
            Dictionary mapping agent IDs to their data
        """
        if bzids is None:
            return self.data
        return {bzid: self.data.get(str(bzid)) for bzid in bzids if str(bzid) in self.data}
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert agent data to a pandas DataFrame.
        
        Returns:
            DataFrame with agent data, with one row per agent
        """
        if not self.data:
            return pd.DataFrame()
            
        # Flatten the data structure for DataFrame conversion
        flattened = []
        for bzid, agent_data in self.data.items():
            flat_agent = {'bzid': bzid}
            # Flatten sources into top-level columns
            for source, data in agent_data.get('sources', {}).items():
                for key, value in data.items():
                    flat_agent[f"{source}_{key}"] = value
            flattened.append(flat_agent)
            
        return pd.DataFrame(flattened)
    
    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Serialize agent data to JSON.
        
        Args:
            filepath: Optional path to save the JSON file. If None, returns JSON string.
            
        Returns:
            JSON string if filepath is None, otherwise None
        """
        output = {
            'metadata': self.metadata,
            'data': self.data
        }
        
        if filepath is None:
            return json.dumps(output, indent=2, default=str)
        
        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Agent data saved to {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error saving agent data to {filepath}: {e}")
            raise
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'AgentDataStore':
        """
        Create an AgentDataStore instance from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            New AgentDataStore instance with data from the file
        """
        filepath = Path(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            store = cls(data.get('data', {}))
            store.metadata = data.get('metadata', {})
            return store
            
        except Exception as e:
            logger.error(f"Error loading agent data from {filepath}: {e}")
            raise
    
    def merge(self, other: 'AgentDataStore') -> None:
        """
        Merge another AgentDataStore into this one.
        
        Args:
            other: Another AgentDataStore instance to merge from
        """
        for bzid, agent_data in other.data.items():
            if bzid in self.data:
                # Merge sources
                self.data[bzid]['sources'].update(agent_data.get('sources', {}))
                # Update metadata
                self.data[bzid]['metadata']['updated_at'] = datetime.utcnow().isoformat()
                self.data[bzid]['metadata']['sources'] = list(set(
                    self.data[bzid]['metadata'].get('sources', []) +
                    agent_data.get('metadata', {}).get('sources', [])
                ))
            else:
                # Add new agent
                self.data[bzid] = agent_data
        
        # Update global metadata
        self.metadata['last_updated'] = datetime.now(timezone.utc).isoformat()
        if 'source_files' in other.metadata:
            self.metadata['source_files'].extend(other.metadata['source_files'])
            self.metadata['source_files'] = list(set(self.metadata['source_files']))


# Global instance for convenience
agent_store = AgentDataStore()

def save_agent_data(filepath: Union[str, Path], data: Optional[Dict] = None) -> None:
    """
    Save agent data to a JSON file.
    
    Args:
        filepath: Path to save the JSON file
        data: Optional data to save. If None, uses the global agent_store.
    """
    store = agent_store if data is None else AgentDataStore(data)
    store.to_json(filepath)

def load_agent_data(filepath: Union[str, Path]) -> AgentDataStore:
    """
    Load agent data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        AgentDataStore instance with the loaded data
    """
    return AgentDataStore.from_json(filepath)
