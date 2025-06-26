"""Test fixtures for region contact data."""
import pandas as pd
import pytest

@pytest.fixture
def sample_region_contact():
    """Return sample region contact data for testing."""
    data = {
        'Region': ['MP', 'Tamil Nadu & Kerala', 'Gujarat'],
        'Manager': [
            'Aswinsatheesh.work@gmail.com',
            'Aswinsatheesh.work@gmail.com',
            'Aswinsatheesh.work@gmail.com'
        ],
        'Name': ['Aswin', 'Aswin', 'Aswin']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_region_contact():
    """Return an empty region contact DataFrame with correct columns."""
    columns = ['Region', 'Manager', 'Name']
    return pd.DataFrame(columns=columns)
