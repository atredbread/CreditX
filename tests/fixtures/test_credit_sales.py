"""Test fixtures for credit sales data."""
import pandas as pd
import pytest
from datetime import datetime

@pytest.fixture
def sample_credit_sales():
    """Return sample credit sales data for testing."""
    data = {
        'DATE': [
            datetime(2025, 6, 1),
            datetime(2025, 6, 1),
            datetime(2025, 6, 1)
        ],
        'account': [28115985, 28372265, 27602131],
        'GMV': [5800.0, 1699.95, 1020.0],
        'tin': ['7AB67HK6', '7A46MMDV', '7AHH4PNE']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_credit_sales():
    """Return an empty credit sales DataFrame with correct columns."""
    columns = ['DATE', 'account', 'GMV', 'tin']
    return pd.DataFrame(columns=columns)
