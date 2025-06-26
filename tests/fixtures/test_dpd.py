"""Test fixtures for DPD (Days Past Due) data."""
import pandas as pd
import pytest

@pytest.fixture
def sample_dpd():
    """Return sample DPD data for testing."""
    data = {
        'Anchor': ['REDBUS', 'REDBUS', 'REDBUS'],
        'Phone': [9885777379, 9885777379, 9885777379],
        'Bzid': [24939241, 24939241, 24939241],
        'Username': ['RAHAMTHULLA SHAIK', 'RAHAMTHULLA SHAIK', 'RAHAMTHULLA SHAIK'],
        'Business Name': ['SRT travels - Rayachoti', 'SRT travels - Rayachoti', 'SRT travels - Rayachoti'],
        'Dpd': [3, 2, 1],
        'Pos': [5379.30, 5379.30, 5379.30]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_dpd():
    """Return an empty DPD DataFrame with correct columns."""
    columns = ['Anchor', 'Phone', 'Bzid', 'Username', 'Business Name', 'Dpd', 'Pos']
    return pd.DataFrame(columns=columns)
