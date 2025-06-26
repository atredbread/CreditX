"""Test fixtures for credit agents data."""
import pandas as pd
import pytest

@pytest.fixture
def sample_credit_agents():
    """Return sample credit agents data for testing."""
    data = {
        'Bzid': [42403057, 25104349, 25305319],
        'Phone': [9320137288, 9227160271, 8885580880],
        'Credit Line Setup Co': ['4-5-2023, 7:42 PM', '26-4-2023, 7:31 PM', '26-4-2023, 8:40 PM'],
        'Approval Amount': [100000, 34000, 100000],
        'Credit Limit': [100000, 34000, 100000],
        'Credit Line Balance': [100000.0, 34000.0, 33370.99],
        'Status': ['D', 'D', 'D']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_credit_agents():
    """Return an empty credit agents DataFrame with correct columns."""
    columns = [
        'Bzid', 'Phone', 'Credit Line Setup Co', 'Approval Amount',
        'Credit Limit', 'Credit Line Balance', 'Status'
    ]
    return pd.DataFrame(columns=columns)
