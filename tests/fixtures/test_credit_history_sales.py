"""Test fixtures for credit history sales data."""
import pandas as pd
import pytest

@pytest.fixture
def sample_credit_history_sales():
    """Return sample credit history sales data for testing."""
    data = {
        'Account': [25125208, 28499610, 13810786],
        'Jan Total GMV - Year24 Sales GMV': [247262, 223279, 413916],
        'Jan Credit Gmv': [8663, 129258, 353787],
        '%Jan Consumption': [0.035, 0.5789, 0.8547],
        'Feb Total GMV - Year24': [312328, 106395, 320612],
        'Feb Credit Gmv': [20045, 18639, 290120],
        '%Feb consumtion': [0.0642, 0.1752, 0.9049],
        'Mar Total GMV - Year24': [760886, 164273, 273848],
        'Mar Credit Gmv': [255745, 0, 236811],
        '%Mar Consumtion': [0.3361, 0.0, 0.8648],
        'Apr Total GMV - Year24': [877220, 283910, 340894],
        'Apr Credit Gmv': [167912, 0, 42072],
        '%Apr Consumtion': [0.1914, 0.0, 0.1234],
        'May Total GMV - Year24': [849623, 302165, 532597],
        'May Credit Gmv': [84957, 0, 0],
        '%May Total GMV - Year24 Consumption': [0.1, 0.0, 0.0],
        'June Total GMV - Year24': [400360, 168483, 262338],
        'June Credit Gmv': [6928, 0, 2835],
        '%Jun Total GMV - Year24 consumtion': [0.0173, 0.0, 0.0108],
        'July Total GMV - Year24': [347233, 119172, 274636],
        'July Credit Gmv': [28278, 0, 1678],
        '% Jul Total GMV - Year24 consumption': [0.0814, 0.0, 0.0061],
        'Aug Total GMV - Year24': [160477, 148787, 286227],
        'Aug Credit Gmv': [0, 0, 0],
        '% Aug Total GMV - Year24 consumption': [0.0, 0.0, 0.0],
        'Sep Total GMV - Year24': [341403.8, 153521.1, 268432.7],
        'Sep Credit Gmv': [115035.5, 0.0, 1774.5],
        '% Sep Total GMV - Year24 consumption': [0.3369, 0.0, 0.0066],
        'Oct Total GMV - Year24': [245044.4, 198267.1, 271862.8],
        'Oct Credit Gmv': [149103.9, 0.0, 0.0],
        '% oct Total GMV - Year24 consumption': [0.6085, 0.0, 0.0],
        'Nov Total GMV - Year24': [827497, 301908, 255248],
        'Nov Credit Gmv': [477138.9, 0.0, 0.0],
        '% Nov Total GMV - Year24 consumption': [0.5766, 0.0, 0.0],
        'Dec Total GMV - Year24': [611169, 253447, 284814],
        'Dec Credit Gmv': [439028, 0, 0],
        '% Dec Total GMV - Year24 consumption': [0.7183, 0.0, 0.0],
        'Jan Total GMV - Year25': [458674, 226523, 319249],
        'Jan Credit Gmv.1': [322444, 0, 0],
        '% Jan Total GMV - Year25 consumption': [0.703, 0.0, 0.0],
        'Feb Total GMV - Year25': [388310, 228508, 236825],
        'Feb Credit Gmv.1': [147399, 0, 0],
        '% Feb Total GMV - Year25 consumption': [0.3796, 0.0, 0.0],
        'Mar Total GMV - Year25': [223093, 89740, 108210],
        'Mar Credit Gmv.1': [64794, 0, 0],
        '% Mar Total GMV - Year25 consumption': [0.2904, 0.0, 0.0],
        'Apr Total GMV - Year25': [507822, 520917, 351687],
        'Apr Credit Gmv.1': [222337, 0, 0],
        '% Apr Total GMV - Year25 consumption': [0.4378, 0.0, 0.0],
        'May Total GMV - Year25': [341265, 249222, 239615],
        'May Credit Gmv.1': [121208, 0, 0],
        '% May Total GMV - Year25 consumption': [0.3552, 0.0, 0.0],
        'June Total GMV - Year25': [324456.6, 63996.15, 98008.61],
        'JuneCredit Gmv': [118592.2, 0.0, 0.0],
        '% May Total GMV - Year25 consumption.1': [0.3655, 0.0, 0.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_credit_history_sales():
    """Return an empty credit history sales DataFrame with correct columns."""
    columns = [
        'Account', 'Jan Total GMV - Year24 Sales GMV', 'Jan Credit Gmv', 
        '%Jan Consumption', 'Feb Total GMV - Year24', 'Feb Credit Gmv', 
        '%Feb consumtion', 'Mar Total GMV - Year24', 'Mar Credit Gmv', 
        '%Mar Consumtion', 'Apr Total GMV - Year24', 'Apr Credit Gmv', 
        '%Apr Consumtion', 'May Total GMV - Year24', 'May Credit Gmv', 
        '%May Total GMV - Year24 Consumption', 'June Total GMV - Year24', 
        'June Credit Gmv', '%Jun Total GMV - Year24 consumtion', 
        'July Total GMV - Year24', 'July Credit Gmv', 
        '% Jul Total GMV - Year24 consumption', 'Aug Total GMV - Year24', 
        'Aug Credit Gmv', '% Aug Total GMV - Year24 consumption', 
        'Sep Total GMV - Year24', 'Sep Credit Gmv', 
        '% Sep Total GMV - Year24 consumption', 'Oct Total GMV - Year24', 
        'Oct Credit Gmv', '% oct Total GMV - Year24 consumption', 
        'Nov Total GMV - Year24', 'Nov Credit Gmv', 
        '% Nov Total GMV - Year24 consumption', 'Dec Total GMV - Year24', 
        'Dec Credit Gmv', '% Dec Total GMV - Year24 consumption', 
        'Jan Total GMV - Year25', 'Jan Credit Gmv.1', 
        '% Jan Total GMV - Year25 consumption', 'Feb Total GMV - Year25', 
        'Feb Credit Gmv.1', '% Feb Total GMV - Year25 consumption', 
        'Mar Total GMV - Year25', 'Mar Credit Gmv.1', 
        '% Mar Total GMV - Year25 consumption', 'Apr Total GMV - Year25', 
        'Apr Credit Gmv.1', '% Apr Total GMV - Year25 consumption', 
        'May Total GMV - Year25', 'May Credit Gmv.1', 
        '% May Total GMV - Year25 consumption', 'June Total GMV - Year25', 
        'JuneCredit Gmv', '% May Total GMV - Year25 consumption.1'
    ]
    return pd.DataFrame(columns=columns)
