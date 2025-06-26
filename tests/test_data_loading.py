"""
Test script to verify data loading functionality of the Credit Health Engine.
"""
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.credit_health_engine import CreditHealthEngine

def main():
    print("Testing Credit Health Engine Data Loading...")
    print("-" * 50)
    
    # Initialize the engine
    engine = CreditHealthEngine()
    
    # Load the data
    print("\nLoading data...")
    success = engine.load_data()
    
    if not success:
        print("\n❌ Failed to load data. Please check the logs for details.")
        return
    
    print("\n✅ Data loaded successfully!")
    print("\nLoaded datasets:")
    print("-" * 50)
    
    # Display information about each loaded dataset
    for file_name, df in engine.data.items():
        print(f"\n📊 {file_name}")
        print(f"   Shape: {df.shape}")
        print("   First few columns:", ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else ""))
        print(f"   Total rows: {len(df):,}")
    
    print("\n✅ Data loading test completed!")

if __name__ == "__main__":
    main()
