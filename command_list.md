typologies = list(scenario_config.keys())
print(typologies)

import pandas as pd
import numpy as np

df3=pd.read_parquet('df3.parquet')

# Convert 'ref_date' to datetime
df3['ref_date'] = pd.to_datetime(df3['ref_date'], format='%Y-%m-%d', errors='coerce')

df3['transaction_date'] = pd.to_datetime(df3['transaction_date'], format='%Y-%m-%d', errors='coerce')

# Sort in ascending order of date
df3 = df3.sort_values(by='ref_date', ascending=True)

# Optional: Reset index after sorting
#df3 = df3.reset_index(drop=True)


import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURATION
# -----------------------------
TRANSACTIONS_FILE = "transactions.csv"   # or .parquet
PARTY_FILE = "party.csv"
SCENARIO_CONFIG_FILE = "scenario_config.json"
OUT_FILE = "scenario_config.json"

# Column names
COL_TX_ID = "transaction_key"     #"tx_id"
COL_PARTY = "party_key"
COL_TX_TYPE = "tx_type"           # e.g., 'CREDIT'/'DEBIT'
COL_AMOUNT = "acct_currency_amount"    #"amount"
COL_TS = "transaction_date"     #"timestamp"
COL_COUNTRY = "counterparty_country"
COL_DIRECTION = "trans_direction"       #"direction"       # optional: 'IN'/'OUT'
COL_SUSP_FLAG = "str_flag"    #"str_flag"   #"suspicious_flag"  # boolean
COL_ENTITY_TYPE = "party_type_cd"    #"entity_type"
COL_IS_PEP = "is_pep"
#COL_NATIONALITY = "nationality"

# Time window for statistics
WINDOW_DAYS = 1800

# Thresholds
MICRO_THRESHOLD = 500.0
BURST_MULTIPLIER = 3.0
MIN_TXS_FOR_STATS = 10
LOCAL_CODES = ['SG']

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def safe_load_csv_or_parquet(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def ensure_datetime(df, col):
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def ensure_positive_amounts(s):
    s = pd.to_numeric(s, errors='coerce')
    return s.fillna(0.0)

def old_compute_stats(series):
    series = series.dropna()
    return {
        "mu": float(series.mean()) if not series.empty else None,
        "sigma": float(series.std(ddof=1)) if len(series)>1 else 0.0,
        "min": float(series.min()) if not series.empty else None,
        "max": float(series.max()) if not series.empty else None,
        "median": float(series.median()) if not series.empty else None,
        "count": len(series)
    }



def compute_stats(series):
    # 1. Clean missing data
    series = series.dropna()
    if series.empty:
        return {"mu": None, "sigma": 0.0, "min": None, "max": None, "median": None, "count": 0}

    # 2. Outlier Removal via IQR (1.5x scale is standard for 2026 AML calibration)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    # Define boundaries: anything outside this range is an outlier
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 3. Filtered Dataset (Removing dust and whales)
    # Ensure we stay above 0 for financial logic
    filtered_series = series[(series >= max(0.01, lower_bound)) & (series <= upper_bound)]
    
    # Fallback: if filtering empties the set, use original (safety for small datasets)
    if filtered_series.empty:
        filtered_series = series

    return {
        "mu": float(filtered_series.mean()),
        "sigma": float(filtered_series.std(ddof=1)) if len(filtered_series) > 1 else 0.0,
        "min": float(filtered_series.min()),
        "max": float(filtered_series.max()),
        "median": float(filtered_series.median()),
        "count": len(filtered_series)
    }


def fit_lognormal_params(amounts):
    arr = np.array([a for a in amounts if a>0])
    if len(arr)<2:
        return None, None
    logs = np.log(arr)
    return float(np.mean(logs)), float(np.std(logs, ddof=1))



def interarrival_stats(times):
    if len(times)<2:
        return {"gap_mean_hours": None, "gap_sigma_hours": None}
    diffs = times.sort_values().diff().dt.total_seconds().dropna()/3600.0
    return {"gap_mean_hours": float(diffs.mean()), "gap_sigma_hours": float(diffs.std(ddof=1) if len(diffs)>1 else 0.0)}

def compute_bursts_per_client(tx_df):
    df = tx_df.copy()
    df['day'] = df[COL_TS].dt.floor('d')
    daily = df.groupby([COL_PARTY,'day']).size().rename('count').reset_index()
    per_party_stats = daily.groupby(COL_PARTY)['count'].agg(['mean','median']).reset_index().rename(columns={'mean':'mean_daily','median':'median_daily'})
    merged = daily.merge(per_party_stats, on=COL_PARTY, how='left')
    #merged['is_burst'] = merged['count'] > (merged['mean_daily']*BURST_MULTIPLIER)

    # MODIFIED LINE: Add a constant (+1 or +2) to prevent a 0-burst median
    # This ensures that even for low-velocity clients, a 'spike' is detectable.
    # Standard 2026 practice is (Mean * Multiplier) + Standard Deviation or Constant
    merged['is_burst'] = merged['count'] > (merged['mean_daily'] * BURST_MULTIPLIER + 1)

    bursts = merged.groupby(COL_PARTY).agg(n_bursts=('is_burst','sum'), burst_mean_size=('count','mean')).reset_index()
    #n_bursts_base = int(np.nanmedian(bursts['n_bursts'])) if not bursts.empty else 0
    burst_size_base = float(np.nanmedian(bursts.loc[bursts['n_bursts']>0,'burst_mean_size']) if (bursts['n_bursts']>0).any() else bursts['burst_mean_size'].median()) if not bursts.empty else 0
    #return {"n_bursts_base": n_bursts_base, "burst_size_base": burst_size_base}

    # 1. Calculate Empirical Bursts from our 2.8M rows
    
    empirical_median = np.nanmedian(bursts['n_bursts']) if not bursts.empty else 0

    # 2. DYNAMIC CALCULATION: 1 burst per quarter (90 days) of observation
    # This automatically scales with your WINDOW_DAYS
    date_range = (tx_df[COL_TS].max() - tx_df[COL_TS].min()).days
    dynamic_floor = int(date_range / 90) 

    # 3. FINAL CALIBRATION: Take the highest of the three (Empirical, Dynamic, or Absolute 1)
    # This ensures your 'Velocity Spike' typology is NEVER zero.

    # 2026 Production Standard: Clamp the base between a logical floor and ceiling
    # Floor of 1: Ensures the typology is never 'empty'
    # Ceiling of 12: Prevents 'Velocity' from becoming 'Constant Noise'
    
    # 1. Take the HIGHER of our real data or the dynamic floor (e.g., max(0, 4) = 4)
    # This ensures we don't drop back to 1 if the population is quiet.
    base_calc = max(dynamic_floor, empirical_median)

    # 2. Apply a Floor of 1 (Absolute Minimum) and a Ceiling of 12 (Absolute Maximum)
    # This prevents the value from being 0, but also caps it at a realistic 12.
    n_bursts_base = int(max(1, min(12, base_calc)))

    
    return {
        "n_bursts_base": n_bursts_base, 
        "burst_size_base": burst_size_base
    }


def fraction_offshore(tx_df, country_col=COL_COUNTRY, local_codes=LOCAL_CODES):
    total = len(tx_df)
    if total==0:
        return 0.0
    offshore = tx_df[~tx_df[country_col].isin(local_codes)]
    return float(len(offshore)/total)

def inflow_outflow_ratio_stats(tx_df):
    df = tx_df.copy()
    if COL_DIRECTION in df.columns:
        df['direction_norm'] = df[COL_DIRECTION].str.upper().map(lambda x: 'IN' if x in ['IN','CREDIT'] else 'OUT')
    else:
        df['direction_norm'] = df[COL_TX_TYPE].str.upper().map(lambda x: 'IN' if 'CREDIT' in x else 'OUT')
    grouped = df.groupby([COL_PARTY,'direction_norm'])[COL_AMOUNT].sum().unstack(fill_value=0.0)
    grouped['inflow'] = grouped.get('IN',0.0)
    grouped['outflow'] = grouped.get('OUT',0.0)
    grouped['ratio'] = grouped.apply(lambda r: (r['inflow']/r['outflow']) if r['outflow']>0 else np.nan, axis=1)
    ratios = grouped['ratio'].replace([np.inf,-np.inf], np.nan).dropna()
    if ratios.empty:
        return {"ratio_mean": None,"ratio_std": None}
    return {"ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std(ddof=1) if len(ratios)>1 else 0.0)}

def monthly_deviation_stats(tx_df):
    df = tx_df.copy()
    df['month'] = df[COL_TS].dt.to_period('M')
    monthly = df.groupby([COL_PARTY,'month'])[COL_AMOUNT].sum().reset_index()
    stats=[]
    for party, g in monthly.groupby(COL_PARTY):
        vals = g[COL_AMOUNT].values
        #if len(vals)<2: continue
        if len(vals)<3: continue
        mean = np.mean(vals)
        if mean==0: continue
        deviations = vals/mean
        stats.append({"party":party,"dev_mean":float(np.mean(deviations)),"dev_std":float(np.std(deviations,ddof=1)),
                      "dev_max":float(np.max(deviations)),"dev_min":float(np.min(deviations))})
    if len(stats)==0:
        return {"monthly_deviation_high": None,"monthly_deviation_low": None,
                "monthly_deviation_alt_high": None,"monthly_deviation_alt_low": None}
    #devs = pd.DataFrame(stats)
    devs = pd.DataFrame(stats).fillna(0.0)

    vols = devs['dev_std']
    
    return {
        #"monthly_deviation_high": float(devs['dev_mean'].quantile(0.90)),
        #"monthly_deviation_low": float(devs['dev_mean'].quantile(0.10)),
        #"monthly_deviation_alt_high": float(devs['dev_std'].quantile(0.75)),
        #"monthly_deviation_alt_low": float(devs['dev_std'].quantile(0.25))
        #"monthly_deviation_high": float(devs['dev_std'].quantile(0.90)),
        #"monthly_deviation_low": float(devs['dev_std'].quantile(0.10)),
        
        # How much do the "extreme" months vary?
        #"monthly_deviation_alt_high": float(devs['dev_max'].quantile(0.75)),
        #"monthly_deviation_alt_low": float(devs['dev_min'].quantile(0.25))

    
        # High: Use 80th percentile instead of 90th to avoid outliers
        #"monthly_deviation_high": float(devs['dev_std'].quantile(0.80)),
        
        # Low: Use the Median (0.50) to represent a "typical" stable business
        #"monthly_deviation_low": float(devs['dev_std'].quantile(0.50)),
        
        # Alt-High: Use a lower quantile for the "Max Spike"
        #"monthly_deviation_alt_high": float(devs['dev_max'].quantile(0.60)),
        
        # Alt-Low: Ensure this isn't near zero by using a higher percentile
        #"monthly_deviation_alt_low": float(devs['dev_min'].quantile(0.40))

        # HIGH: The most extreme volatility (Top 5%)
        "monthly_deviation_high": float(vols.quantile(0.95)),
        
        # ALT HIGH: High volatility, but more common (Top 25%)
        "monthly_deviation_alt_high": float(vols.quantile(0.75)),
        
        # ALT LOW: Low volatility (Bottom 25%)
        "monthly_deviation_alt_low": float(vols.quantile(0.25)),

        # LOW: Extremely stable/flat activity (Bottom 5%)
        "monthly_deviation_low": float(vols.quantile(0.05))

        
    }

        

def old_hops_proxy(tx_df):
    res = tx_df.groupby(COL_PARTY)[COL_COUNTRY].nunique().rename('n_counterparty_countries').reset_index()
    return {"n_hops_base": int(np.nanmedian(res['n_counterparty_countries'])) if not res.empty else 0}

def hops_proxy(tx_df):
    # 2026 Logic: Use unique COUNTERPARTIES (not countries) to estimate potential hops
    # Assuming you have a column for counterparty (e.g., account_key or transaction_key)
    res = tx_df.groupby(COL_PARTY)[COL_COUNTRY].nunique().rename('n_countries').reset_index()
    
    # Calculate base hops: If they talk to 1 country, they likely have 2-3 hops in a chain
    # We apply a floor of 3 because 'Layering' by definition requires a chain.
    n_hops_raw = int(np.nanmedian(res['n_countries'])) if not res.empty else 0
    
    # 2026 Production Standard: Layering must have a chain of at least 3 hops
    return {"n_hops_base": max(3, n_hops_raw + 1)}



def update_json_field(path_list, value, scn):
    node = scn
    for key in path_list[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path_list[-1]] = value

def infer_tx_type(direction):
    if direction.upper() == "INCOMING":
        return "CREDIT"     # client receives money
    elif direction.upper() == "OUTGOING":
        return "DEBIT"      # client sends money
    else:
        return "UNKNOWN"

df3[COL_TX_TYPE]=df3['trans_direction'].apply(infer_tx_type)

# -----------------------------
# LOAD DATA
# -----------------------------
tx = df3  #safe_load_csv_or_parquet(TRANSACTIONS_FILE)
party = final_features_selected   #safe_load_csv_or_parquet(PARTY_FILE)

tx['counterparty_country']="SG"

tx[COL_PARTY] = tx[COL_PARTY].astype(str)
party[COL_PARTY] = party[COL_PARTY].astype(str)

tx = ensure_datetime(tx, COL_TS)
tx[COL_AMOUNT] = ensure_positive_amounts(tx[COL_AMOUNT])

# Filter suspicious transactions
#suspicious_party_keys = set(party.loc[party[COL_SUSP_FLAG]==True, COL_PARTY])

privatebanking_party_keys = set(
    party.loc[
        #(party[COL_SUSP_FLAG] == True) &
        (party['pbtpc_segment'].str.strip().str.upper().isin(['TPC', 'PB'])),
        COL_PARTY
    ]
)



#tx_window = tx[tx[COL_PARTY].isin(privatebanking_party_keys) & (tx[COL_TS]>=tx[COL_TS].max()-timedelta(days=WINDOW_DAYS))].copy()
tx_window = tx[ tx[COL_PARTY].isin(privatebanking_party_keys) ].copy()

print(f"Private Banking clients: {len(privatebanking_party_keys)}, private banking transactions: {len(tx_window)}")


# Load scenario config
#with open(SCENARIO_CONFIG_FILE,'r') as f:
    #scenario = json.load(f)

import json

# --- Load JSON ---
with open("scenario_config.json", "r") as f:
    #scenario_config = json.load(f)
    scenario = json.load(f)

# -----------------------------
# CALIBRATION
# -----------------------------
# --- 1. Structuring ---
#credit_mask = tx_window[COL_TX_TYPE].str.upper().fillna("").str.contains("CREDIT") if COL_TX_TYPE in tx_window.columns else (tx_window[COL_DIRECTION].str.upper()=='IN')

# CORRECTED: Use regex '|' for OR and map to your specific INCOMING/OUTGOING labels
credit_mask = (
    tx_window[COL_TX_TYPE].str.upper().fillna("").str.contains("CREDIT") 
    if COL_TX_TYPE in tx_window.columns 
    else tx_window[COL_DIRECTION].str.upper().isin(['INCOMING'])
)


credit_tx = tx_window[credit_mask]
#struct_stats = compute_stats(credit_tx[COL_AMOUNT])
struct_stats = compute_stats(tx_window[COL_AMOUNT])
gaps = interarrival_stats(credit_tx[COL_TS].sort_values())
credit_counts = credit_tx.groupby(COL_PARTY).size()
n_credits_base = int(np.nanmedian(credit_counts)) if len(credit_counts)>0 else scenario.get('structuring',{}).get('n_credits_base',8)
n_credits_min = int(np.nanpercentile(credit_counts,5)) if len(credit_counts)>0 else scenario.get('structuring',{}).get('n_credits_min',5)

update_json_field(['structuring','n_credits_base'], n_credits_base, scenario)
update_json_field(['structuring','n_credits_min'], n_credits_min, scenario)
update_json_field(['structuring','gap_mean'], round(gaps.get('gap_mean_hours',1.0),4), scenario)
update_json_field(['structuring','gap_sigma'], round(gaps.get('gap_sigma_hours',0.7),4), scenario)
update_json_field(['structuring','amt_mu'], round(struct_stats['mu'],2), scenario)
update_json_field(['structuring','amt_sigma'], round(struct_stats['sigma'],2), scenario)
update_json_field(['structuring','amt_min'], float(struct_stats['min']), scenario)
update_json_field(['structuring','amt_max'], float(struct_stats['max']), scenario)


# Update biz_flag_non_nexus

# --- 2. Biz Flag Non-Nexus Calibration ---
# Filter for typical business activity (Non-PEP, Corporate/SmallBiz if available)
biz_mask = tx_window[COL_TX_TYPE].str.upper().fillna("").str.contains("CREDIT|DEBIT")
biz_tx = tx_window[biz_mask]

# Calculate Real Velocity (Gap Days)
# We sort by party and date, calculate differences, and convert seconds to days
biz_sorted = biz_tx.sort_values([COL_PARTY, COL_TS])
biz_sorted['gap'] = biz_sorted.groupby(COL_PARTY)[COL_TS].diff().dt.total_seconds() / (3600 * 24)
real_gap_days = float(biz_sorted['gap'].median()) if not biz_sorted['gap'].dropna().empty else 2.0


# Calculate Volume and Amount stats
#biz_counts = biz_tx.groupby(COL_PARTY).size()
#n_tx_base = int(np.nanmedian(biz_counts)) if len(biz_counts) > 0 else 6


# FIX for Non-Nexus frequency: Use only CREDITS to drop the count from 88 to ~10
biz_counts = tx_window[tx_window[COL_TX_TYPE].str.upper().str.contains("CREDIT")].groupby(COL_PARTY).size()
n_tx_base = int(np.nanmedian(biz_counts)) if len(biz_counts) > 0 else 6


biz_stats = compute_stats(biz_tx[COL_AMOUNT])

# Update JSON fields using your existing update_json_field helper
#update_json_field(['biz_flag_non_nexus','tx_gap_days'], round(real_gap_days, 2), scenario)

# Apply a floor of 0.1 to ensure the simulation always advances in time
update_json_field(['biz_flag_non_nexus','tx_gap_days'], max(0.1, round(real_gap_days, 2)), scenario)



update_json_field(['biz_flag_non_nexus','n_tx_base'], n_tx_base, scenario)
update_json_field(['biz_flag_non_nexus','amt_min'], round(float(biz_stats.get('min', 500)), 2), scenario)
update_json_field(['biz_flag_non_nexus','amt_max'], round(float(biz_stats.get('median', 5000) * 2), 2), scenario)

# Update Micro-transaction params based on real median (2026 'Smurfing' Thresholds)
if 'micro_params' in scenario.get('biz_flag_non_nexus', {}):
    micro_max = float(biz_stats.get('median', 1000) * 0.15) # 15% of median is the 2026 micro-cap
    update_json_field(['biz_flag_non_nexus','micro_params','amt_max'], round(micro_max, 2), scenario)


print(f"[INFO] Calibration Complete for 'biz_flag_non_nexus'.")
print(f"[INFO] Inferred tx_gap_days: {real_gap_days:.2f}")



# Micro transactions
# 2026 Calibration: Remove dust (<$1) and whale outliers (top 2%) for micro-stability

# --- FIX 1: The Primary Micro Data ---
micro_tx = tx_window[(tx_window[COL_AMOUNT] >= 1.0) & (tx_window[COL_AMOUNT] <= MICRO_THRESHOLD)]
if len(micro_tx) > 20:
    micro_tx = micro_tx[micro_tx[COL_AMOUNT] <= micro_tx[COL_AMOUNT].quantile(0.98)]
if micro_tx.empty:
    micro_tx = pd.DataFrame({COL_AMOUNT: [1.0, float(MICRO_THRESHOLD)]})

#micro_tx = tx_window[tx_window[COL_AMOUNT]<=MICRO_THRESHOLD]
micro_alpha, micro_beta = 1.5,3.5
if len(micro_tx)>=5:
    minv,maxv = micro_tx[COL_AMOUNT].min(), micro_tx[COL_AMOUNT].max()
    if maxv>minv:
        scaled = (micro_tx[COL_AMOUNT]-minv)/(maxv-minv+1e-9)
        m,v = scaled.mean(), scaled.var(ddof=0)
        if v>0 and m*(1-m)>v:
            common = m*(1-m)/v-1.0
            micro_alpha = max(0.1,float(m*common))
            micro_beta = max(0.1,float((1-m)*common))
micro_amt_min = float(micro_tx[COL_AMOUNT].min()) if len(micro_tx)>0 else struct_stats['min']
micro_amt_max = float(micro_tx[COL_AMOUNT].max()) if len(micro_tx)>0 else struct_stats['max']
micro_multiplier = float(max(1,(len(micro_tx)/max(1,len(tx_window)))*10))

update_json_field(['structuring','micro_alpha'], round(micro_alpha,3), scenario)
update_json_field(['structuring','micro_beta'], round(micro_beta,3), scenario)
update_json_field(['structuring','micro_multiplier'], round(micro_multiplier,3), scenario)
update_json_field(['structuring','micro_amt_min'], micro_amt_min, scenario)
update_json_field(['structuring','micro_amt_max'], micro_amt_max, scenario)

# --- 2. Velocity Spike ---
burst_info = compute_bursts_per_client(tx_window)
update_json_field(['velocity_spike','n_bursts_base'], burst_info['n_bursts_base'], scenario)
update_json_field(['velocity_spike','burst_size_base'], burst_info['burst_size_base'], scenario)
update_json_field(['velocity_spike','amt_mu'], round(struct_stats['mu'],2), scenario)
update_json_field(['velocity_spike','amt_sigma'], round(struct_stats['sigma'],2), scenario)
update_json_field(['velocity_spike','amt_min'], struct_stats['min'], scenario)
update_json_field(['velocity_spike','amt_max'], struct_stats['max'], scenario)

# --- SUPERSEDED: 2026 Calibration to remove dust (<$1) and whale outliers (top 2%) ---
# --- FIX 2: The Variation Micro Data ---
# 1. Primary Filter: Dust floor ($1) and Micro Ceiling
micro_tx_vs = tx_window[(tx_window[COL_AMOUNT] >= 1.0) & (tx_window[COL_AMOUNT] <= MICRO_THRESHOLD)]

# 2. Whale Removal: Statistical clipping
if len(micro_tx_vs) > 20:
    micro_tx_vs = micro_tx_vs[micro_tx_vs[COL_AMOUNT] <= micro_tx_vs[COL_AMOUNT].quantile(0.98)]

# 3. GUARANTEED FALLBACK: If the filter resulted in an empty set, 
# create a single-row "Synthetic anchor" to prevent billion-dollar leakage.
if micro_tx_vs.empty:
    micro_tx_vs = pd.DataFrame({COL_AMOUNT: [1.0, float(MICRO_THRESHOLD)]})


# --- REMAINDER OF CODE: UNCHANGED ---
#micro_tx_vs = tx_window[tx_window[COL_AMOUNT]<=MICRO_THRESHOLD]

micro_alpha_vs, micro_beta_vs = 1.5,4.0
if len(micro_tx_vs)>=5:
    minv,maxv = micro_tx_vs[COL_AMOUNT].min(),micro_tx_vs[COL_AMOUNT].max()
    if maxv>minv:
        scaled = (micro_tx_vs[COL_AMOUNT]-minv)/(maxv-minv+1e-9)
        m,v = scaled.mean(), scaled.var(ddof=0)
        if v>0 and m*(1-m)>v:
            common = m*(1-m)/v-1.0
            micro_alpha_vs = max(0.1,float(m*common))
            micro_beta_vs = max(0.1,float((1-m)*common))
update_json_field(['velocity_spike','micro_alpha'], round(micro_alpha_vs,3), scenario)
update_json_field(['velocity_spike','micro_beta'], round(micro_beta_vs,3), scenario)
update_json_field(['velocity_spike','micro_amt_min'], float(micro_tx_vs[COL_AMOUNT].min()) if len(micro_tx_vs)>0 else struct_stats['min'], scenario)
update_json_field(['velocity_spike','micro_amt_max'], float(micro_tx_vs[COL_AMOUNT].max()) if len(micro_tx_vs)>0 else struct_stats['max'], scenario)
update_json_field(['velocity_spike','micro_multiplier'], round(max(1,(len(micro_tx_vs)/max(1,len(tx_window)))*10),3), scenario)

# --- 3. Round Trip ---
mu_log,sigma_log = fit_lognormal_params(tx_window[COL_AMOUNT].tolist())
if mu_log is not None:
    update_json_field(['round_trip','base_amt_logmu'], mu_log, scenario)
    update_json_field(['round_trip','base_amt_logsigma'], sigma_log if sigma_log is not None else 0.3, scenario)

# --- Robust flat_gaps calculation ---
all_gaps = []

# Group by party and compute interarrival times
for party_key, ts in tx_window.groupby(COL_PARTY)[COL_TS]:
    ts_sorted = ts.sort_values()
    diffs_hours = ts_sorted.diff().dt.total_seconds().dropna() / 3600.0  # in hours
    if len(diffs_hours) > 0:
        all_gaps.append(diffs_hours.values)  # always a numpy array

# Concatenate all gaps into a single array
flat_gaps = np.concatenate(all_gaps) if len(all_gaps) > 0 else np.array([])

# Now flat_gaps can be safely used
if flat_gaps.size > 1:
    update_json_field(['round_trip','hop_gap_logmu'], float(np.mean(np.log(flat_gaps + 1e-9))), scenario)
    update_json_field(['round_trip','hop_gap_logsigma'], float(np.std(np.log(flat_gaps + 1e-9), ddof=1)), scenario)
    update_json_field(['round_trip','return_gap_mu'], float(np.mean(flat_gaps)), scenario)
    update_json_field(['round_trip','return_gap_sigma'], float(np.std(flat_gaps, ddof=1)), scenario)

inout = tx_window.copy()
if COL_DIRECTION in inout.columns:
    inout['dir'] = inout[COL_DIRECTION].str.upper()
else:
    inout['dir'] = inout[COL_TX_TYPE].str.upper().map(lambda x: 'IN' if 'CREDIT' in x else 'OUT')
agg = inout.groupby([COL_PARTY,'dir'])[COL_AMOUNT].sum().unstack(fill_value=0.0)

ratios=[]
for _,row in agg.iterrows():
    out = row.get('OUT',0.0)
    inc = row.get('IN',0.0)
    if out>0: ratios.append(min(1.0,inc/out))
if len(ratios)>=3:
    m = float(np.mean(ratios)); v=float(np.var(ratios,ddof=0))
    if v==0: v=m*(1-m)/10
    common = m*(1-m)/v-1.0 if v>0 and m*(1-m)>v else 2.0
    a=max(0.1,m*common); b=max(0.1,(1-m)*common)
    update_json_field(['round_trip','return_frac_a'], round(a,3), scenario)
    update_json_field(['round_trip','return_frac_b'], round(b,3), scenario)
    update_json_field(['round_trip','fee_frac_a'], 1.5, scenario)
    update_json_field(['round_trip','fee_frac_b'], 8.0, scenario)

# --- 4. Layering ---
hop_proxy_res = hops_proxy(tx_window)
update_json_field(['layering','n_hops_base'], hop_proxy_res['n_hops_base'], scenario)
amt_stats_layer = compute_stats(tx_window[COL_AMOUNT])
update_json_field(['layering','amt_min'], float(amt_stats_layer['min']), scenario)
update_json_field(['layering','amt_max'], float(amt_stats_layer['max']), scenario)
gaps_all = interarrival_stats(tx_window[COL_TS].sort_values())
update_json_field(['layering','hop_gap_hours_scale'], round(gaps_all.get('gap_mean_hours',12.0),4), scenario)

# --- 5. Business inflow/outflow ratio ---
#ratio_stats = inflow_outflow_ratio_stats(tx_window)
#if ratio_stats['ratio_mean'] is not None:
    #inflow_mult_high = max(1.0,ratio_stats['ratio_mean'])
    #outflow_mult_high = max(1.0,1.0/ratio_stats['ratio_mean'] if ratio_stats['ratio_mean']>0 else 1.0)
    #update_json_field(['biz_inflow_outflow_ratio','inflow_multiplier_high'], round(inflow_mult_high,3), scenario)
    #update_json_field(['biz_inflow_outflow_ratio','outflow_multiplier_high'], round(outflow_mult_high,3), scenario)
    #update_json_field(['biz_inflow_outflow_ratio','inflow_multiplier_low'], round(max(0.5,inflow_mult_high*0.8),3), scenario)
    #update_json_field(['biz_inflow_outflow_ratio','outflow_multiplier_low'], round(max(0.5,outflow_mult_high*0.8),3), scenario)

# --- 5. Business inflow/outflow ratio (Improved) ---
ratio_stats = inflow_outflow_ratio_stats(tx_window)

if ratio_stats['ratio_mean'] is not None:
    mean = ratio_stats['ratio_mean']
    # Use standard deviation to determine spread, fallback to a default (e.g., 0.25) if std is 0
    std = ratio_stats.get('ratio_std', 0.25) 
    
    # 1. Calculate High Multiplier: Ensure it captures the upper bound of activity
    # Formula: Mean + (1 Std Dev), but at least 1.5 as per your original intent
    inflow_mult_high = max(1.5, mean + std)
    outflow_mult_high = max(1.5, (1.0/mean if mean > 0 else 1.0) + std)

    # 2. Calculate Low Multiplier: Ensure it captures the lower bound
    # Formula: Mean - (1 Std Dev), capped at your intended 0.8 or a hard floor of 0.1
    #inflow_mult_low = min(0.8, max(0.1, mean - std))
    #outflow_mult_low = min(0.8, max(0.1, (1.0/mean if mean > 0 else 1.0) - std))

    # Use 0.5 as the floor for 'Low' instead of 0.1
    # This ensures a substantial gap (1.5 vs 0.5) without "killing" the activity
    inflow_mult_low = min(0.8, max(0.5, mean - std))
    outflow_mult_low = min(0.8, max(0.5, (1.0/mean if mean > 0 else 1.0) - std))
    
    # Update JSON
    update_json_field(['biz_inflow_outflow_ratio','inflow_multiplier_high'], round(inflow_mult_high, 3), scenario)
    update_json_field(['biz_inflow_outflow_ratio','outflow_multiplier_high'], round(outflow_mult_high, 3), scenario)
    update_json_field(['biz_inflow_outflow_ratio','inflow_multiplier_low'], round(inflow_mult_low, 3), scenario)
    update_json_field(['biz_inflow_outflow_ratio','outflow_multiplier_low'], round(outflow_mult_low, 3), scenario)


# --- 6. Monthly volume deviation ---
md = monthly_deviation_stats(tx_window)
for k,v in md.items():
    update_json_field(['biz_monthly_volume_deviation',k], v, scenario)

# --- 7. Business round-tripping ---
#biz_tx = tx_window[tx_window[COL_ENTITY_TYPE].str.lower().isin(['corporation','company','sme','corporate'])] if COL_ENTITY_TYPE in tx_window.columns else tx_window
#biz_tx = tx_window[tx_window[COL_ENTITY_TYPE].str.lower().isin(['indv','corp'])] if COL_ENTITY_TYPE in tx_window.columns else tx_window
#if len(biz_tx)>0:
    #biz_amt_stats = compute_stats(biz_tx[COL_AMOUNT])
    #update_json_field(['biz_round_trip','amt_mu'], biz_amt_stats['mu'], scenario)
    #update_json_field(['biz_round_trip','amt_sigma'], biz_amt_stats['sigma'], scenario)


# --- Calculate Business Amount Stats ---
biz_tx = tx_window[tx_window[COL_ENTITY_TYPE].str.lower().isin(['indv','corp'])] if COL_ENTITY_TYPE in tx_window.columns else tx_window

if not biz_tx.empty:
    biz_stats = compute_stats(biz_tx[COL_AMOUNT])
    
    # Map to the specific keys used in Part 2 of your production logic
    #update_json_field(['biz_round_tripping', 'structuring_amt_mu'], biz_stats['mu'], scenario)
    #update_json_field(['biz_round_tripping', 'structuring_amt_sigma'], biz_stats['sigma'], scenario)

    # Squeeze logic to keep simulation stable
    update_json_field(['biz_round_tripping', 'structuring_amt_mu'], min(biz_stats['mu'], 100000), scenario)
    update_json_field(['biz_round_tripping', 'structuring_amt_sigma'], min(biz_stats['sigma'], biz_stats['mu'] * 0.5), scenario)
    
    update_json_field(['biz_round_tripping', 'amt_min'], biz_stats['min'], scenario)

    #update_json_field(['biz_round_tripping', 'amt_max'], biz_stats['max'], scenario)      
    #update_json_field(['biz_round_tripping', 'amt_max'], min(biz_stats['max'], 5000000), scenario) # Cap at 5M
    
    stable_max = min(biz_stats['max'], 5000000)
    update_json_field(['biz_round_tripping', 'amt_max'], stable_max, scenario)
  
    #update_json_field(['biz_round_tripping', 'structuring_amt_max'], biz_stats['max'], scenario)    
    update_json_field(['biz_round_tripping', 'structuring_amt_max'], stable_max, scenario)


# --- Cycle Frequency ---
avg_tx_count = biz_tx.groupby(COL_PARTY).size().mean()


# --- Gap Analysis from your Robust flat_gaps logic ---
if flat_gaps.size > 1:

    # 1. DYNAMIC CALIBRATION: Find the smallest gap > 0 to determine the shift
    # This prevents hardcoding '24' while ensuring the log-input is always >= 1
    min_observed = np.min(flat_gaps[flat_gaps > 0]) if np.any(flat_gaps > 0) else 1.0
    
    # 2. CALCULATE SHIFT: If min_observed is 0.0001, dynamic_shift becomes 10000
    # This ensures even your smallest observed gap results in ln(1) = 0
    dynamic_shift = 1.0 / min_observed

    # 3. APPLY SHIFT: Transform gaps so that the smallest gap maps to 1.0
    # We use np.maximum to handle any remaining absolute zeros safely
    log_gaps = np.log(np.maximum(1.0, flat_gaps * dynamic_shift))
    
    update_json_field(['biz_round_tripping', 'structuring_gap_mean'], float(np.mean(log_gaps)), scenario)
    update_json_field(['biz_round_tripping', 'structuring_gap_sigma'], float(np.std(log_gaps, ddof=1)), scenario)
    
    
    # 'cycle_gap_hours' is the scale for the exponential distribution in the HOP phase
    #update_json_field(['biz_round_tripping', 'cycle_gap_hours'], float(np.mean(flat_gaps)), scenario)

    # Ensure cycle_gap_hours has a sensible lower bound (e.g., at least 1 hour)
    avg_gap = float(np.mean(flat_gaps))
    update_json_field(['biz_round_tripping', 'cycle_gap_hours'], max(1.0, avg_gap), scenario)

    # Calibrate the base transaction count for structuring bursts
    update_json_field(['biz_round_tripping', 'structuring_tx_base'], max(5, int(avg_tx_count // 4)), scenario)


# --- Hop Proxy Calibration ---
hops_data = hops_proxy(biz_tx)
# Ensure max_hops is grounded in the number of unique counterparty countries observed
update_json_field(['biz_round_tripping', 'max_hops'], max(5, hops_data['n_hops_base'] * 3), scenario)

# --- Cycle Frequency ---
#avg_tx_count = biz_tx.groupby(COL_PARTY).size().mean()
# If the average entity does 20 txs, and structuring_tx_base is 8, 
# then 2 cycles (base) makes sense (2 * 8 = 16).
#update_json_field(['biz_round_tripping', 'n_cycles_base'], max(2, int(avg_tx_count // 10)), scenario)
update_json_field(['biz_round_tripping', 'n_cycles_base'], min(5, int(avg_tx_count // 10)), scenario)

# Calculate how much amounts typically fluctuate between transactions
if len(biz_tx) > 1:
    ratios = biz_tx[COL_AMOUNT].pct_change().dropna() + 1
    # Filter out extreme outliers for a stable multiplier range
    low_bound = float(np.percentile(ratios, 10)) 
    high_bound = float(np.percentile(ratios, 90))
    
    update_json_field(['biz_round_tripping', 'hop_multiplier_low'], round(max(0.1, low_bound), 2), scenario)
    update_json_field(['biz_round_tripping', 'hop_multiplier_high'], round(min(5.0, high_bound), 2), scenario)



# ==========================
# Helper functions
# ==========================
def superoldest_compute_micro_amount_stats(df, amt_col=COL_AMOUNT):     #'amount'
    """Compute mean, std, min, max for micro parameter purposes"""
    return df[amt_col].mean(), df[amt_col].std(), df[amt_col].min(), df[amt_col].max()

def oldest_compute_micro_amount_stats(df, amt_col=COL_AMOUNT):
    """Compute stats specifically for the micro-segment (Retrofit Logic)"""
    micro_tx = df[df[amt_col] <= MICRO_THRESHOLD][amt_col]
    
    if micro_tx.empty:
        # Fallback to general stats if no micro transactions exist
        return df[amt_col].mean(), df[amt_col].std(), 0.0, MICRO_THRESHOLD
        
    return micro_tx.mean(), micro_tx.std(), micro_tx.min(), micro_tx.max()

def old_compute_micro_amount_stats(df, amt_col=COL_AMOUNT, threshold=MICRO_THRESHOLD):
    """Compute stats using a dynamic threshold"""
    micro_tx = df[df[amt_col] <= threshold][amt_col]
    if micro_tx.empty:
        return df[amt_col].mean(), df[amt_col].std(), 0.0, threshold
    return micro_tx.mean(), micro_tx.std(), micro_tx.min(), micro_tx.max()


def compute_micro_amount_stats(df, amt_col=COL_AMOUNT, threshold=MICRO_THRESHOLD):
    """Compute stats using a dynamic threshold and removing dust/whale outliers"""
    # 1. Primary Filter: Keep only transactions between $1.00 and the MICRO_THRESHOLD
    mask = (df[amt_col] >= 1.0) & (df[amt_col] <= threshold)
    micro_tx = df[mask][amt_col]
    
    if micro_tx.empty:
        # 2. SAFE FALLBACK: If no micro-txs exist, do NOT return global billion-dollar stats.
        # Return a logical 1.0 to threshold range for the synthetic generator.
        return 1.0, 0.0, 1.0, float(threshold)
        
    # 3. Success: Return the actual stats of the cleaned micro-range
    return (
        float(micro_tx.mean()), 
        float(micro_tx.std(ddof=1)) if len(micro_tx) > 1 else 0.0, 
        float(micro_tx.min()), 
        float(micro_tx.max())
    )


    
def compute_micro_tx_types(df):
    """Return the unique transaction types observed"""
    return df['tx_type'].dropna().unique().tolist()

def SUPERoldest_compute_alpha_beta(df):
    """Simple proxy for alpha/beta: shape and scale of transaction amount distribution"""
    amt = df[COL_AMOUNT].values             #'amount'
    if len(amt) < 2:
        return 1.0, 1.0
    mu, sigma = np.mean(amt), np.std(amt)
    alpha = max(mu / (sigma + 1e-6), 0.5)
    beta = max(sigma / (mu + 1e-6), 0.5)
    return alpha, beta

def oldest_compute_alpha_beta(df):
    """Formal Method of Moments for Beta Distribution (Retrofit Logic)"""
    # 1. Filter for micro transactions only (Contrasting Logic)
    micro_tx = df[df[COL_AMOUNT] <= MICRO_THRESHOLD][COL_AMOUNT]
    
    # Defaults
    m_alpha, m_beta = 1.5, 3.5 
    
    if len(micro_tx) >= 5:
        minv, maxv = micro_tx.min(), micro_tx.max()
        if maxv > minv:
            # 2. Scale data 0 to 1 (Required for Beta)
            scaled = (micro_tx - minv) / (maxv - minv + 1e-9)
            m, v = scaled.mean(), scaled.var(ddof=0)
            
            # 3. Mathematical Safety Check
            if 0 < v < m * (1 - m):
                common = (m * (1 - m) / v) - 1.0
                m_alpha = max(0.1, float(m * common))
                m_beta = max(0.1, float((1 - m) * common))
                
    return m_alpha, m_beta

def old_compute_alpha_beta(df, threshold=MICRO_THRESHOLD):
    """Compute Alpha/Beta using a dynamic threshold"""
    micro_tx = df[df[COL_AMOUNT] <= threshold][COL_AMOUNT]
    m_alpha, m_beta = 1.5, 3.5 
    if len(micro_tx) >= 5:
        minv, maxv = micro_tx.min(), micro_tx.max()
        if maxv > minv:
            scaled = (micro_tx - minv) / (maxv - minv + 1e-9)
            m, v = scaled.mean(), scaled.var(ddof=0)
            if 0 < v < m * (1 - m):
                common = (m * (1 - m) / v) - 1.0
                m_alpha = max(0.1, float(m * common))
                m_beta = max(0.1, float((1 - m) * common))
    return m_alpha, m_beta


def compute_alpha_beta(df, threshold=MICRO_THRESHOLD):
    """Compute Alpha/Beta using a dynamic threshold with outlier clipping"""
    # 1. Base filter
    micro_tx = df[df[COL_AMOUNT] <= threshold][COL_AMOUNT]
    
    # 2. Outlier Removal: Remove the top 2% of 'Micro' txs to stabilize the Beta curve
    if len(micro_tx) > 20:
        micro_tx = micro_tx[micro_tx <= micro_tx.quantile(0.98)]

    m_alpha, m_beta = 1.5, 3.5 
    if len(micro_tx) >= 5:
        minv, maxv = micro_tx.min(), micro_tx.max()
        if maxv > minv:
            scaled = (micro_tx - minv) / (maxv - minv + 1e-9)
            m, v = scaled.mean(), scaled.var(ddof=0)
            if 0 < v < m * (1 - m):
                common = (m * (1 - m) / v) - 1.0
                m_alpha = max(0.1, float(m * common))
                m_beta = max(0.1, float((1 - m) * common))
    return m_alpha, m_beta    

# ==========================
# Initialize JSON structure
# ==========================
#with open("scenario_config_template.json", "r") as f:
    #scenario_config = json.load(f)

# Typologies to process
#typologies = list(scenario_config.keys())
typologies = list(scenario.keys())

# ==========================
# Fill micro parameters dynamically
# ==========================

# Mapping for 2026 Private Banking risk levels
THRESHOLD_MAP = {
    'structuring': 500.0,
    'velocity_spike': 2500.0,
    'layering': 5000.0,
    'round_trip': 5000.0,
    'biz_monthly_volume_deviation': 100.0,
    'biz_round_tripping': 1000.0,
    'biz_flag_non_nexus': 1000.0,  # <--- Correct and calibrated
    
    # RELEVANT FOR 2026: Calibrates micro-padding for High-Risk PEP flows
    'biz_flag_pep_indonesia': 500.0,
    
    # RELEVANT FOR 2026: Isolates personal-to-corp "padding" from macro flows
    'biz_flag_personal_to_corp': 1000.0    
    
}

for typ in typologies:
    # FAIL-SAFE: Use specific map, or default MICRO_THRESHOLD. 
    # For 'biz_flag' types, we use a huge number to include all data.
    if 'biz_flag' in typ or 'biz_monthly' in typ:
        current_threshold = 1e12 
    else:
        current_threshold = THRESHOLD_MAP.get(typ, MICRO_THRESHOLD)

    # PASS the dynamic threshold to your functions
    micro_amt_mu, micro_amt_sigma, micro_amt_min, micro_amt_max = compute_micro_amount_stats(tx_window, threshold=current_threshold)
    micro_tx_types = compute_micro_tx_types(tx_window)
    micro_alpha, micro_beta = compute_alpha_beta(tx_window, threshold=current_threshold)
    
    #tx_typ = transactions_suspicious[transactions_suspicious['typology'] == typ]
    #if tx_typ.empty:
        #continue
    
    # Compute basic micro parameters
    #micro_amt_mu, micro_amt_sigma, micro_amt_min, micro_amt_max = compute_micro_amount_stats(tx_window)     #tx_typ)
    #micro_tx_types = compute_micro_tx_types(tx_window)           #tx_typ)
    #micro_alpha, micro_beta = compute_alpha_beta(tx_window)      #tx_typ)
    
    # Populate into JSON
    if 'micro_params' in scenario[typ]:
        # Nested micro_params structure
        scenario[typ]['micro_params']['alpha'] = round(micro_alpha, 3)
        scenario[typ]['micro_params']['beta'] = round(micro_beta, 3)
        scenario[typ]['micro_params']['multiplier'] = 1  # Default; can adjust based on business logic
        scenario[typ]['micro_params']['amt_min'] = float(micro_amt_min or 0.0)  #int(micro_amt_min)
        scenario[typ]['micro_params']['amt_max'] = float(micro_amt_max or 0.0)  #int(micro_amt_max)
        scenario[typ]['micro_params']['tx_types'] = micro_tx_types
    else:
        # Flat micro_* keys
        if 'micro_alpha' in scenario[typ]:
            scenario[typ]['micro_alpha'] = round(micro_alpha, 3)
        if 'micro_beta' in scenario[typ]:
            scenario[typ]['micro_beta'] = round(micro_beta, 3)
        if 'micro_multiplier' in scenario[typ]:
            scenario[typ]['micro_multiplier'] = 1  # Default; adjust if needed
        if 'micro_amt_min' in scenario[typ]:
            scenario[typ]['micro_amt_min'] = float(micro_amt_min or 0.0)  #int(micro_amt_min)
        if 'micro_amt_max' in scenario[typ]:
            scenario[typ]['micro_amt_max'] = float(micro_amt_max or 0.0)  #int(micro_amt_max)
        if 'micro_tx_types' in scenario[typ]:
            scenario[typ]['micro_tx_types'] = micro_tx_types

tx_window['country'] = 'SG'

# -----------------------------
# Populate remaining fields from actual suspicious transactions
# -----------------------------
typologies = scenario.keys()

for typ in typologies:
    #if typ not in tx_window['typology'].unique():
        #continue
    #tx_typ = tx_window[tx_window['typology']==typ]

    #if tx_typ.empty:
        #continue

    # -----------------------------
    # Country pools (primary, domestic, offshore)
    # -----------------------------
    countries = tx_window['country'].dropna().unique().tolist()
    if 'country_pool_primary' in scenario[typ]:
        scenario[typ]['country_pool_primary'] = countries
    # Simple split: assume domestic = local country, offshore = others
    if 'domestic_pool' in scenario[typ]:    
        scenario[typ]['domestic_pool'] = [c for c in countries if c in LOCAL_CODES]    #['SG']
    if 'offshore_pool' in scenario[typ]:    
        scenario[typ]['offshore_pool'] = [c for c in countries if c not in scenario[typ]['domestic_pool']]


    # -----------------------------
    # Monthly deviation (biz_monthly_volume_deviation)
    # -----------------------------
    #if typ == 'biz_monthly_volume_deviation':
        #tx_window['month'] = tx_window[COL_TS].dt.to_period('M')    # 'timestamp'
        #monthly_sum = tx_window.groupby([COL_PARTY,'month'])[COL_AMOUNT].sum().reset_index()    #'amount'
        #deviation = monthly_sum.groupby(COL_PARTY)[COL_AMOUNT].std() / (monthly_sum.groupby(COL_PARTY)[COL_AMOUNT].mean() + 1e-9)    # 'amount'
        #scenario[typ]['monthly_deviation_high'] = float(deviation.quantile(0.75))
        #scenario[typ]['monthly_deviation_low'] = float(deviation.quantile(0.25))
        #scenario[typ]['monthly_deviation_alt_high'] = float(deviation.quantile(0.60))
        #scenario[typ]['monthly_deviation_alt_low'] = float(deviation.quantile(0.40))

        
    # -----------------------------
    # Forward amount decay / hop gap / multipliers (layering, round_trip)
    # -----------------------------
    if typ in ['layering']:     #,'round_trip']:
        amounts = tx_window[COL_AMOUNT].values          # 'amount'
        if len(amounts) > 1:
            # decay factor approx: ratio of successive tx amounts
            ratios = amounts[1:] / (amounts[:-1] + 1e-6)
            scenario[typ]['demographics']['forward_amt_decay'] = {
                'default': [float(np.quantile(ratios,0.05)), float(np.quantile(ratios,0.995))],
                'high_risk': [float(np.quantile(ratios,0.05)*0.95), float(np.quantile(ratios,0.995))]
            }
            # hop gap scale
            time_diffs = tx_window[COL_TS].diff().dt.total_seconds().dropna()/3600      #'timestamp'
            scenario[typ]['demographics']['hop_gap_hours_scale_adjustment'] = {
                'corporate': float(time_diffs.median()*1.2),
                'high_risk_offshore': float(time_diffs.median()*0.8)
            }

# --- Category 1: Unbiased Population Stats (Private Banking) ---
# Use the full window (both Directions/Types) to capture total behavior
population_tx = tx_window  

# Compute generic statistics for the entire population
pop_stats = compute_stats(population_tx[COL_AMOUNT])


# Update JSON with generic 'population' field (unbiased anchor)
update_json_field(['population','amt_mu'], round(pop_stats['mu'], 2), scenario)
update_json_field(['population','amt_sigma'], round(pop_stats['sigma'], 2), scenario)
update_json_field(['population','amt_min'], float(pop_stats['min']), scenario)
update_json_field(['population','amt_max'], float(pop_stats['max']), scenario)


def old_calibrate_population_timing(df_window):
    """Calculates population-level timing parameters from real data in 2026."""
    
    if df_window.empty:
        return 0.0, {"mean_days": 0.0, "sigma_days": 0.0, "median_days": 0.0, "distribution": "log-normal"}

    # 1. Calculate Daily Lambda (Poisson)
    total_tx = len(df_window)
    n_parties = df_window[COL_PARTY].nunique()
    
    # Correction: Use +1 to include the full day and prevent division by zero
    date_diff = (df_window[COL_TS].max() - df_window[COL_TS].min()).days
    date_range = max(date_diff, 1) 
    
    population_lambda = total_tx / (n_parties * date_range)

    # 2. Calculate Inter-arrival Stats
    # IMPORTANT: sort_values must include COL_TS to get correct chronological gaps
    # .dt.days correctly returns 0 for same-day transactions
    all_gaps = (
        df_window.sort_values([COL_PARTY, COL_TS])
        .groupby(COL_PARTY)[COL_TS]
        .diff()
        .dt.days
        .dropna()
    )
    
    inter_stats = {
        "mean_days": float(all_gaps.mean()) if not all_gaps.empty else 0.0,
        "sigma_days": float(all_gaps.std(ddof=1)) if len(all_gaps) > 1 else 0.0,
        "median_days": float(all_gaps.median()) if not all_gaps.empty else 0.0,
        "distribution": "log-normal" # Standard for anchor modeling
    }

    return population_lambda, inter_stats



# --- 2. REFINED TEMPORAL ANCHOR ---
def calibrate_population_timing(df_window):
    if df_window.empty:
        return 0.1, {"mean_days": 1.0, "sigma_days": 1.0, "median_days": 1.0, "distribution": "log-normal"}

    # Calculate inter-arrival gaps in days
    all_gaps = (
        df_window.sort_values([COL_PARTY, COL_TS])
        .groupby(COL_PARTY)[COL_TS]
        .diff().dt.days.dropna()
    )
    
    # 2026 Production Standard: Anchor the median at 0.5 to prevent simulation 'clumping'
    inter_stats = {
        "mean_days": float(all_gaps.mean()) if not all_gaps.empty else 1.0,
        "sigma_days": float(all_gaps.std(ddof=1)) if len(all_gaps) > 1 else 0.5,
        "median_days": max(0.5, float(all_gaps.median())) if not all_gaps.empty else 0.5,
        "distribution": "log-normal"
    }
    
    # Standard Poisson calculation
    date_range = max((df_window[COL_TS].max() - df_window[COL_TS].min()).days, 1)
    pop_lambda = len(df_window) / (df_window[COL_PARTY].nunique() * date_range)
    
    return pop_lambda, inter_stats



# --- EXECUTION & STORAGE ---
pop_lambda, pop_inter_arrival = calibrate_population_timing(tx_window)

# Store in JSON following your preferred "temporal" rank
update_json_field(['temporal', 'poisson_lambda_daily'], pop_lambda, scenario)
update_json_field(['temporal', 'inter_arrival', 'mean_days'], pop_inter_arrival['mean_days'], scenario)
update_json_field(['temporal', 'inter_arrival', 'sigma_days'], pop_inter_arrival['sigma_days'], scenario)
update_json_field(['temporal', 'inter_arrival', 'median_days'], pop_inter_arrival['median_days'], scenario)
update_json_field(['temporal', 'inter_arrival', 'distribution'], pop_inter_arrival['distribution'], scenario)


# -----------------------------
# SAVE CALIBRATED SCENARIO
# -----------------------------
with open(OUT_FILE,'w') as f:
    json.dump(scenario,f,indent=2)

print(f"Calibrated scenario saved to {OUT_FILE}")

typologies = scenario.keys()
print(typologies)

df3=pd.read_parquet('df3.parquet')

# Convert 'ref_date' to datetime
df3['ref_date'] = pd.to_datetime(df3['ref_date'], format='%Y-%m-%d', errors='coerce')

df3['transaction_date'] = pd.to_datetime(df3['transaction_date'], format='%Y-%m-%d', errors='coerce')

# Sort in ascending order of date
df3 = df3.sort_values(by='ref_date', ascending=True)

# Optional: Reset index after sorting
#df3 = df3.reset_index(drop=True)

with open("scenario_config.json", "r") as f:
    scenario_config = json.load(f)

scenario_config
