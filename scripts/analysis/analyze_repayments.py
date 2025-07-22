"""
Credit Health Intelligence Engine - Repayment Analysis Script
Version: 1.0.0

Strictly implements repayment-based agent scoring as per CREDIT_RISK_FEATURES.md and docs/rule book.md.
No extra features. Data source: source_data/.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ========== Constants & Config ==========
REQUIRED_COLUMNS = {
    'agent_id': 'Bzid',
    'repayment_date': 'Customer Repayment Date',
    'repayment_amount': 'Customer Repayment Amount',
    'principal_repaid': 'Customer Principle Repaid'
}

# ========== Data Loader ==========
def load_repayment_data(source_dir: str) -> pd.DataFrame:
    """
    Loads and validates repayment data from the source_data directory.
    Returns a cleaned DataFrame with required columns.
    """
    # Find repayment report CSV
    files = os.listdir(source_dir)
    csv_file = next((f for f in files if f.lower().startswith('repayment') and f.lower().endswith('.csv')), None)
    if not csv_file:
        logger.error('No repayment report CSV found in source_data/')
        sys.exit(1)
    path = os.path.join(source_dir, csv_file)
    df = pd.read_csv(path)
    # Rename columns to standard names
    col_map = {v: k for k, v in REQUIRED_COLUMNS.items() if v in df.columns}
    df = df.rename(columns=col_map)
    # Ensure all required columns exist
    for k in REQUIRED_COLUMNS:
        if k not in df.columns:
            logger.error(f'Missing required column: {REQUIRED_COLUMNS[k]}')
            sys.exit(1)
    # Clean and convert
    df['repayment_amount'] = pd.to_numeric(df['repayment_amount'], errors='coerce')
    df['principal_repaid'] = pd.to_numeric(df['principal_repaid'], errors='coerce')
    df['repayment_date'] = pd.to_datetime(df['repayment_date'], errors='coerce')
    df['agent_id'] = df['agent_id'].astype(str)
    df = df.dropna(subset=['repayment_amount', 'principal_repaid', 'repayment_date', 'agent_id'])
    logger.info(f"Loaded {len(df)} repayment transactions.")
    return df

# ========== Metrics Calculation ==========
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all required repayment metrics and risk scores per agent.
    Returns a DataFrame with all required columns.
    """
    grouped = df.groupby('agent_id')
    metrics = []
    # First pass for min/max and for normalized columns
    totals, principals, counts, avgs, avg_principals, ratios = [], [], [], [], [], []
    stds, data_qualities, interest_paids, freqs = [], [], [], []
    for agent, group in grouped:
        total = group['repayment_amount'].sum()
        principal = group['principal_repaid'].sum()
        count = len(group)
        avg = total / count if count else 0
        avg_principal = principal / count if count else 0
        ratio = principal / total if total else 0
        std = group['repayment_amount'].std(ddof=0)
        first_tx = group['repayment_date'].min()
        last_tx = group['repayment_date'].max()
        # Data quality: fraction of non-null/valid transactions
        data_quality = group[['repayment_amount', 'principal_repaid', 'repayment_date']].notnull().all(axis=1).mean()
        interest_paid = total - principal
        # Repayment frequency: transactions per month
        days = (last_tx - first_tx).days or 1
        months = max(days / 30.44, 1)
        freq = count / months
        totals.append(total)
        principals.append(principal)
        counts.append(count)
        avgs.append(avg)
        avg_principals.append(avg_principal)
        ratios.append(ratio)
        stds.append(std)
        data_qualities.append(data_quality)
        interest_paids.append(interest_paid)
        freqs.append(freq)
    # Min/max for normalization
    def safe_minmax(arr):
        return (min(arr), max(arr)) if arr else (0, 1)
    min_max = {
        'total': safe_minmax(totals),
        'principal': safe_minmax(principals),
        'count': safe_minmax(counts),
        'avg': safe_minmax(avgs),
        'avg_principal': safe_minmax(avg_principals),
        'ratio': safe_minmax(ratios),
        'std': safe_minmax(stds),
        'data_quality': safe_minmax(data_qualities),
        'interest_paid': safe_minmax(interest_paids),
        'freq': safe_minmax(freqs),
    }
    # Second pass for normalized, weighted score, risk, confidence
    for i, (agent, group) in enumerate(grouped):
        total = totals[i]
        principal = principals[i]
        count = counts[i]
        avg = avgs[i]
        avg_principal = avg_principals[i]
        ratio = ratios[i]
        std = stds[i]
        data_quality = data_qualities[i]
        interest_paid = interest_paids[i]
        freq = freqs[i]
        first_tx = group['repayment_date'].min()
        last_tx = group['repayment_date'].max()
        def norm(val, minv, maxv):
            return 0.0 if maxv == minv else (val - minv) / (maxv - minv)
        total_n = norm(total, *min_max['total'])
        principal_n = norm(principal, *min_max['principal'])
        count_n = norm(count, *min_max['count'])
        avg_n = norm(avg, *min_max['avg'])
        avg_principal_n = norm(avg_principal, *min_max['avg_principal'])
        ratio_n = norm(ratio, *min_max['ratio'])
        std_n = norm(std, *min_max['std'])
        data_quality_n = norm(data_quality, *min_max['data_quality'])
        interest_paid_n = norm(interest_paid, *min_max['interest_paid'])
        freq_n = norm(freq, *min_max['freq'])
        # Weighted sum for credit_health_score (example weights, can be adjusted)
        score = (
            0.20 * total_n +
            0.15 * principal_n +
            0.10 * count_n +
            0.10 * avg_n +
            0.10 * avg_principal_n +
            0.10 * ratio_n +
            0.05 * std_n +
            0.10 * data_quality_n +
            0.05 * freq_n
        ) * 100
        # Confidence: based on data quality and transaction count
        confidence = min(1.0, 0.5 * data_quality + 0.5 * (count_n))
        # Risk category: assign by percentile (top 20% = Low, middle 60% = Medium, bottom 20% = High)
        risk_category = None  # Will be assigned after all scores are computed
        metrics.append({
            'agent_id': agent,
            'total_repayment_amount': total,
            'avg_repayment_amount': avg,
            'std_repayment_amount': std,
            'total_principal_repaid': principal,
            'avg_principal_repaid': avg_principal,
            'transaction_count': count,
            'first_transaction': first_tx,
            'last_transaction': last_tx,
            'data_quality': data_quality,
            'principal_ratio': ratio,
            'interest_paid': interest_paid,
            'repayment_frequency': freq,
            'total_repayment_amount_norm': total_n,
            'total_principal_repaid_norm': principal_n,
            'transaction_count_norm': count_n,
            'avg_repayment_amount_norm': avg_n,
            'avg_principal_repaid_norm': avg_principal_n,
            'principal_ratio_norm': ratio_n,
            'data_quality_norm': data_quality_n,
            'credit_health_score': round(score, 2),
            'confidence': round(confidence, 2),
            'risk_category': risk_category
        })
    # Ensure output columns order
    columns = [
        'agent_id',
        'total_repayment_amount',
        'avg_repayment_amount',
        'std_repayment_amount',
        'total_principal_repaid',
        'avg_principal_repaid',
        'transaction_count',
        'first_transaction',
        'last_transaction',
        'data_quality',
        'principal_ratio',
        'interest_paid',
        'repayment_frequency',
        'total_repayment_amount_norm',
        'total_principal_repaid_norm',
        'transaction_count_norm',
        'avg_repayment_amount_norm',
        'avg_principal_repaid_norm',
        'principal_ratio_norm',
        'data_quality_norm',
        'credit_health_score',
        'confidence',
        'risk_category'
    ]
    # Assign risk categories by percentile after all scores are computed
    import numpy as np
    df_metrics = pd.DataFrame(metrics)
    if not df_metrics.empty:
        scores = df_metrics['credit_health_score']
        # Calculate percentiles
        low_cutoff = np.percentile(scores, 80)
        high_cutoff = np.percentile(scores, 20)
        def assign_risk(score):
            if score >= low_cutoff:
                return 'Low'
            elif score < high_cutoff:
                return 'High'
            else:
                return 'Medium'
        df_metrics['risk_category'] = scores.apply(assign_risk)
    return df_metrics[columns]


# ========== Main CLI ==========
def main():
    """
    CLI entry point. Loads data, computes metrics, merges DPD, and saves report.
    """
    logger.info("Starting repayment analysis...")
    df = load_repayment_data(os.path.join(os.path.dirname(__file__), '../../source_data'))
    metrics_df = calculate_metrics(df)
    # Load and merge DPD data
    dpd_path = os.path.join(os.path.dirname(__file__), '../../source_data/DPD.xlsx')
    if os.path.exists(dpd_path):
        dpd_df = pd.read_excel(dpd_path)
        dpd_df['Bzid'] = dpd_df['Bzid'].astype(str)
        # Use most recent DPD per agent (if multiple)
        dpd_latest = dpd_df.sort_values('Dpd', ascending=False).drop_duplicates('Bzid', keep='first')[['Bzid', 'Dpd']]
        dpd_latest = dpd_latest.rename(columns={'Bzid': 'agent_id', 'Dpd': 'current_dpd'})
        metrics_df = metrics_df.merge(dpd_latest, on='agent_id', how='left')
    else:
        logger.warning('DPD.xlsx not found; DPD features will default to best case.')
        metrics_df['current_dpd'] = None
    # DPD subscore mapping
    def dpd_subscore(dpd):
        if pd.isna(dpd):
            return 1.0
        try:
            d = float(dpd)
        except Exception:
            return 1.0
        if d == 0:
            return 1.0
        elif d < 30:
            return 0.7
        elif d < 60:
            return 0.4
        else:
            return 0.1
    metrics_df['DPD_subscore'] = metrics_df['current_dpd'].apply(dpd_subscore)
    # Normalize repayment score
    metrics_df['repayment_score_norm'] = metrics_df['credit_health_score'] / 100.0
    # Final combined score
    metrics_df['final_score'] = 0.7 * metrics_df['repayment_score_norm'] + 0.3 * metrics_df['DPD_subscore']
    # Assign risk categories by percentile of final_score
    if not metrics_df.empty:
        scores = metrics_df['final_score']
        low_cutoff = np.percentile(scores, 80)
        high_cutoff = np.percentile(scores, 20)
        def assign_risk(score):
            if score >= low_cutoff:
                return 'Low'
            elif score < high_cutoff:
                return 'High'
            else:
                return 'Medium'
        metrics_df['risk_category'] = scores.apply(assign_risk)
    # Write to reports/ folder
    reports_dir = os.path.join(os.path.dirname(__file__), '../../reports')
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'repayment_analysis.csv')
    metrics_df.to_csv(out_path, index=False)
    logger.info(f"Saved repayment analysis report to {out_path}")
    print(metrics_df.head())

if __name__ == "__main__":
    main()