"""Test fixtures for sales data."""
import pandas as pd
import pytest
from datetime import datetime

@pytest.fixture
def sample_sales_data():
    """Return sample sales data for testing."""
    data = {
        'creationtime': [
            datetime(2025, 6, 1),
            datetime(2025, 6, 1),
            datetime(2025, 6, 1)
        ],
        'account': [28115985, 20692257, 28372265],
        'organizationname': [
            'National Travels Belgaum',
            'Shree Tours & Travels Karad 9021250999',
            'Prisha Tours and Travels'
        ],
        'status': ['BOOKED', 'BOOKED', 'BOOKED'],
        'TotalSeats': [4, 1, 1],
        'GMV': [5800.0, 2280.0, 1699.95],
        'AgentCommission': [580.0, 0.0, 80.95],
        'city': ['Belagavi', 'Karad', 'Mumbai'],
        'State': ['Karnataka', 'Pune', 'Mumbai'],
        'Region': ['Karnataka', 'MH+Goa', 'MH+Goa'],
        'Check': [False, False, False],
        'Ro Name': ['Sohail fazal', 'Bajirao yewale', 'Rajkumarreddy Ranjolkar'],
        'RM Name': ['Shivaraj', 'Shivaraj', 'Shriram S']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_sales_data():
    """Return an empty sales data DataFrame with correct columns."""
    columns = [
        'creationtime', 'account', 'organizationname', 'status', 'TotalSeats',
        'GMV', 'AgentCommission', 'city', 'State', 'Region', 'Check',
        'Ro Name', 'RM Name'
    ]
    return pd.DataFrame(columns=columns)
