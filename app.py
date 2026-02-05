# ================================================================
# FULL AML SYNTHETIC + HYBRID GENERATOR + CATEGORICAL SUPPORT
# WITH AML-TYPOLOGY BIASED CATEGORICAL FEATURES
# ================================================================

from scipy.stats import truncnorm
from scipy.stats import genpareto
import optuna


# Initialize the study once
# direction="maximize" means it will eventually target high Recall/Scores
#study = optuna.create_study(direction="maximize", load_if_exists=True)

# 1. Create the study (run this once outside your if/elif logic)
study = optuna.create_study(direction="maximize") 

# --- Inside your if/elif block ---

# 2. "Ask" for a new trial object
trial = study.ask()     


import json
import numpy as np
import pandas as pd
from scipy.stats import genpareto

def transactions_demographic_selected(df, selected_field):
    frequency_series = df[selected_field].value_counts(normalize=True, dropna=False)
    dynamic_weight_map = frequency_series.to_dict()

    outcomes = list(dynamic_weight_map.keys())
    weights = list(dynamic_weight_map.values())   # no rounding needed unless required

    return outcomes, weights

import random

def preload_transactions_distributions(df, list_of_fields):
    transactions_map = {}

    for fld in list_of_fields:
        outcomes, weights = transactions_demographic_selected(df, fld)

        transactions_map[fld] = {
            "outcomes": outcomes,
            "weights": weights,
        }

    return transactions_map

fields = [ "party_type_cd", "non_nexus_flag", "acct_currency_cd", "trans_type_cd", "channel_type_cd", "transaction_strings", "cashier_order_flag", "trans_direction" ]
txn_map = preload_transactions_distributions(df3, fields)

def sample_from_transactions(transactions_map, key):
    trans = transactions_map[key]
    return random.choices(trans["outcomes"], weights=trans["weights"])[0]

#Save dictionary:

import json

with open("transactions_map.json", "w") as f:
    json.dump(txn_map, f)


#Load dictionary:

with open("transactions_map.json") as f:
    txn_map = json.load(f)
    

# ----------------- Ensure global SCENARIO_CONFIG exists -----------------
#SCENARIO_CONFIG = globals().get('SCENARIO_CONFIG', {})

from scipy import stats
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scipy.stats import truncnorm



# ---------------- Helper pools & constants ----------------
_HIGH_RISK = ["BVI","CAY","PANAMA","AE","HK","VG"]
_ASIA = ["SG","HK","TH","VN","ID","MY","PH"]
_MAJOR = ["US","GB","EU"]
_CASINO = ["CASINO_HK","CASINO_MG","CASINO_NZ","CASINO_PH"]
_INVEST = ["INV_VEH_1","INV_VEH_2","INV_FUND_A"]
_OFFSHORE = ["CAY","BVI","BERMUDA","PANAMA","LUX","SWITZERLAND"]

# ---------------- AML-typology biased categorical helpers ----------------
def _biased_party_type():
    return random.choices(["CORP", "INDV"], weights=[0.3, 0.7])[0]

def _biased_non_nexus_flag():
    return random.choices([True, False], weights=[0.1, 0.9])[0]

def _biased_acct_currency():
    currencies = ["SGD", "USD", "EUR", "GBP", "JPY", "AUD", "HKD", "THB"]
    weights = [0.70, 0.15, 0.08, 0.03, 0.02, 0.005, 0.005, 0.005]
    return random.choices(currencies, weights=weights)[0]

def _biased_trans_type_cd():
    codes = ["045", "111", "046", "049", "133", "109", "007", "005", "112",
             "001", "110", "003", "053", "004", "006", "008", "137", "047", "002", "013"]
    base_counts = [2360643, 733020, 383048, 213654, 126115, 98093, 78857, 46799,
                   43338, 27974, 12595, 6859, 4347, 3901, 1793, 1716, 929, 268, 140, 39]
    norm = np.array(base_counts) / sum(base_counts)
    # Example risk boost: slightly upweight 046,049 (wire/trade-like)
    risk_boost = np.array([1.0 if code in ["046", "049"] else 1.0 for code in codes])
    weights = norm * risk_boost
    return random.choices(codes, weights=weights)[0]

def _biased_channel_type():
    channels = ["GCSP_Online", "FTSP", "NETS", "BRANCH", "ATM", "ACH", "ATM+", "BTM+", "Others"]
    weights = [0.40, 0.20, 0.10, 0.10, 0.07, 0.05, 0.03, 0.02, 0.03]
    return random.choices(channels, weights=weights)[0]

def _biased_transaction_strings():
    return "DQSP/0/N/0/1/5/5/RNSP/#/#/#/#/#/#/#"

def _biased_cashier_order_flag():
    return random.choices([True, False], weights=[0.05, 0.95])[0]

def _biased_trans_direction():
    return random.choices(["OUTGOING", "INCOMING"], weights=[0.9, 0.1])[0]

def _biased_days_prior_review():
    return random.randint(0, 15)

# ---------------- Neutral / baseline helpers ----------------
def _rand_transaction_key(acct, t): return f"{acct}_{int(t.timestamp())}_{random.randint(100,999)}"
def _rand_balance(): return float(np.random.uniform(5_000, 800_000))
def _rand_party_country(): return random.choice(_ASIA + _HIGH_RISK + _MAJOR)
def _rand_party_key(): return f"PARTY_{np.random.randint(1000,9999)}"

def _old_rand_local_currency(amount, currency=None):
    rates = {"SGD":1.0,"USD":1.35,"HKD":0.17,"EUR":1.5}
    rate = rates.get(currency, 1.0)
    return round(amount * rate,2)

def _rand_local_currency(amount, currency=None):
    rates = {"SGD": 1.0, "USD": 1.35, "HKD": 0.17, "EUR": 1.5}

    if currency is None:
        # Define currencies and their corresponding weights (probabilities as percentages)
        currencies = list(rates.keys())
        weights = [96, 4/3, 4/3, 4/3] # Sums to 100
        
        # Randomly select a currency based on weights
        selected_currency = random.choices(currencies, weights=weights, k=1)[0]
        rate = rates[selected_currency]
        #print(f"No currency specified. Randomly selected {selected_currency} with {weights[currencies.index(selected_currency)]}% probability.")
    else:
        rate = rates.get(currency, 1.0)
        
    return round(amount * rate, 2)



# ---- Aliases for older helper names referenced in mock-data creation ----
# These preserve compatibility with the earlier branch that used _rand_* names

_rand_party_type = sample_from_transactions(txn_map, "party_type_cd")  #_biased_party_type
_rand_non_nexus_flag = sample_from_transactions(txn_map, "non_nexus_flag")  #_biased_non_nexus_flag
_rand_acct_currency = sample_from_transactions(txn_map, "acct_currency_cd")  #_biased_acct_currency
_rand_trans_type_cd = sample_from_transactions(txn_map, "trans_type_cd")  #_biased_trans_type_cd
_rand_channel_type = sample_from_transactions(txn_map, "channel_type_cd")  #_biased_channel_type
_rand_transaction_strings = sample_from_transactions(txn_map, "transaction_strings")  #_biased_transaction_strings
_rand_cashier_order_flag = sample_from_transactions(txn_map, "cashier_order_flag")  #_biased_cashier_order_flag
_rand_days_prior_review = _biased_days_prior_review
_rand_trans_direction = sample_from_transactions(txn_map, "trans_direction")  #_biased_trans_direction

def _rand_transaction_date(t): return t.date()


import json




# ======================================================
# MAIN GENERATOR
# ======================================================

def generate_events_from_params(acct, ref_date, anchor, intensity, params, party_key=None, opt_overrides=None):
    """
    Config-driven event generator for all AML scenarios.
    All numeric constants, thresholds, and country pools are read from scenario_config dict.
    Python logic now only handles iteration, random draws, and arithmetic transformations.
    """

    # 1. DEFINE TRIAL ONCE HERE (Top Level)
    # This makes .suggest_float and .suggest_int available to all elif typologies below
    from types import SimpleNamespace
    trial = SimpleNamespace(
        suggest_float = lambda name, low, high: (low + high) / 2,
        suggest_int   = lambda name, low, high: int((low + high) / 2)
    )


    # FIRST MOMENT: Ensure opt_overrides is a dictionary, even if None was passed
    if opt_overrides is None:
        opt_overrides = {}
    
    
    scenario_type = params.get("scenario_type", params.get("name", "structuring"))
    rows = []

    if party_key is None:
        party_key = _rand_party_key()

    intensity = max(0.1, float(intensity))

    
    # ------------------ Helper functions ------------------
    def _assemble_tx(acct, party_key, ref_date, event_time, amt, tx_type, config, label, pools=None):
    #def _assemble_tx(acct, party_key, ref_date, event_time, amt, tx_type, config, pools=None, label):

        # ref_date: The fixed review date for the client (eg 2025-02-28)
        # event_time: The specific time the suspicious event occurs (near the anchor)

        # Mapping from trans_type_cd to direction
        trans_direction_map = {
            "INCOMING": ["001", "002", "005", "006", "053", "109", "110", "137"],
            "OUTGOING": ["003", "004", "007", "008", "013", "045", "046", "047", "049", "111", "112", "133"]
        }

        
        #party_type_lookup = synthetic_parties_df.drop_duplicates(subset='party_key', keep='last').set_index('party_key')['party_type_cd'].to_dict()
        #non_nexus_lookup = synthetic_parties_df.drop_duplicates(subset='party_key', keep='last').set_index('party_key')['non_nexus_flag'].to_dict()

        pools = pools or {}
        cp_country_iso = random.choice(pools.get('offshore_pool', _HIGH_RISK + _ASIA + _MAJOR + ["LOCAL"])
                                       if random.random() < pools.get('offshore_prob', 0.5)
                                       else pools.get('domestic_pool', _ASIA + _MAJOR + _HIGH_RISK))
        party_country_iso = random.choice(pools.get('country_pool_primary', _HIGH_RISK + _ASIA + _MAJOR + ["LOCAL"]))

        # 1. TRANSACTION DATE LOGIC
        # Business Sense: The event_time IS the transaction time.
        # 'Noise' should be minor (minutes/hours) or omitted if event_time is already randomized.
        transaction_date = event_time

        # 2. DAYS PRIOR REVIEW LOGIC
        # This must be the distance from the FIXED review date
        days_prior_review = (ref_date - transaction_date).days

        # --- Generate trans_type_cd ---
        #trans_type_cd = sample_from_transactions(txn_map, "trans_type_cd")  #_biased_trans_type_cd()  
        
        # --- Determine direction based on trans_type_cd ---
        #if trans_type_cd in trans_direction_map["INCOMING"]:
            #trans_direction = "INCOMING"
        #elif trans_type_cd in trans_direction_map["OUTGOING"]:
            #trans_direction = "OUTGOING"
        #else:
            #trans_direction = random.choice(["INCOMING","OUTGOING"])

        # 2. Define list of typologies that require strict directional alignment
        strict_direction_scenarios = ["biz_inflow_outflow_ratio", "biz_round_tripping"]
        
        # --- RECTIFIED 2026 DIRECTIONAL LOGIC ---
        # If this is a flow-through scenario, we MUST force the code to match the tx_mode
        if scenario_type in strict_direction_scenarios:
            if tx_type == "CREDIT":
                trans_type_cd = np.random.choice(trans_direction_map["INCOMING"])
                trans_direction = "INCOMING"
            else:
                trans_type_cd = np.random.choice(trans_direction_map["OUTGOING"])
                trans_direction = "OUTGOING"
        #else:
            # Standard logic for other scenarios
            #trans_type_cd = sample_from_transactions(txn_map, "trans_type_cd")
            #if trans_type_cd in trans_direction_map["INCOMING"]:
                #trans_direction = "INCOMING"
            #elif trans_type_cd in trans_direction_map["OUTGOING"]:
                #trans_direction = "OUTGOING"


        else:
            # 1. Determine the 'Ground Truth' Direction from the Typology Input
            # We map the high-level tx_type to a binary direction for validation
            intended_dir = "INCOMING" if tx_type in ["CREDIT"] else "OUTGOING"
            target_codes = trans_direction_map[intended_dir]

            # 2. REJECTION SAMPLING: Try to find a legacy code that matches intent
            match_found = False
            for _ in range(20):
                candidate_code = sample_from_transactions(txn_map, "trans_type_cd")
                if candidate_code in target_codes:
                    trans_type_cd = candidate_code
                    trans_direction = intended_dir
                    match_found = True
                    break
            
            # 3. FALLBACK: If 20 tries fail, pick a valid code from the correct pool
            if not match_found:
                trans_type_cd = np.random.choice(target_codes)
                trans_direction = intended_dir
        
        
        return [
            acct,
            party_key,
            ref_date,     # Column 3: Always the fixed client review date
            tx_type,
            amt,
            cp_country_iso,
            round(_rand_balance(), 2),
            party_country_iso,
            
            label,     # label_suspicious
            True,  # injected_flag

            'Placeholder for party_type_cd',   #party_type_lookup.get(party_key, sample_from_transactions(txn_map, "party_type_cd")   ), #_biased_party_type()),
            'Placeholder for non_nexus_flag',  #non_nexus_lookup.get(party_key,  sample_from_transactions(txn_map, "non_nexus_flag")  ), #_biased_non_nexus_flag()),            
            
            sample_from_transactions(txn_map, "acct_currency_cd"), #_biased_acct_currency(),
            _rand_transaction_key(acct, transaction_date),    # Use transaction_date for key
            trans_type_cd,   
            sample_from_transactions(txn_map, "channel_type_cd"), #_biased_channel_type(),
            sample_from_transactions(txn_map, "transaction_strings"), #_biased_transaction_strings(),
            sample_from_transactions(txn_map, "cashier_order_flag"), #_biased_cashier_order_flag(),
            _rand_local_currency(amt, sample_from_transactions(txn_map, "acct_currency_cd")  ),  #_biased_acct_currency()),
            days_prior_review,    
            trans_direction,   
            transaction_date.strftime('%Y-%m-%d %H:%M:%S')    
        ]


    # THE UNIVERSAL FIX: 
    # Making 'label' optional with a default (=None) 
    # allows your existing callbacks to "skip" its position safely.
    def _generate_micro_tx(base_time, base_amt, config, label=None, pools=None, ref_date=None, **kwargs):

    
    # VALID 2026 FIX: Assign a default value to 'label'
    #def _generate_micro_tx(base_time, base_amt, config, label=None, pools=None, ref_date=None):

    #def _generate_micro_tx(base_time, base_amt, config, label, pools=None, ref_date=None):
    #def _generate_micro_tx(base_time, base_amt, config, pools=None, ref_date=None, label):
        micro_rows = []
        micro_count = max(0, int(np.random.beta(config.get('micro_alpha', 1.2),
                                                config.get('micro_beta', 3.0))
                                 * config.get('micro_multiplier', 3) * intensity))
        for _ in range(micro_count):
            t_micro = base_time + timedelta(hours=np.random.uniform(config.get('micro_time_min', 0.1),
                                                                    config.get('micro_time_max', 6.0)))
            amt_micro = round(np.random.uniform(config.get('micro_amt_min', 10),
                                               config.get('micro_amt_max', 100)), 2)
            #tx_type_micro = random.choice(config.get('micro_tx_types', ["DEBIT","CREDIT","P2P"]))
            tx_type_micro = random.choice(config.get('micro_tx_types', ["DEBIT","CREDIT"]))
            micro_rows.append(_assemble_tx(acct, party_key, ref_date, t_micro, amt_micro, tx_type_micro, config, label, pools=pools))
        return micro_rows



    def extreme_value_theory(tx_window, scenario_type=None, threshold_pct=95, confidence_level=0.999, sample_range=(0.5, 0.95), sample_n=5):
    
        import json
        import numpy as np
        import pandas as pd
        from scipy.stats import genpareto
        
        # --- 1. CORE EVT CALIBRATION LOGIC ---
        
        def calibrate_gpd_params(series, threshold_pct, mode='high'):
            series = series.dropna()
            if series.empty:
                return {"threshold_u": 0.0, "sigma_scale": 0.0, "xi_shape": 0.0, "count_above_u": 0}
            
            if mode == 'low':
                # Inversion: Treat 1/amount as the tail (tiny amounts become large values)
                data = 1 / (series + 0.01)
            else:
                data = series
        
            u = np.percentile(data, threshold_pct)
            exceedances = data[data > u] - u
            
            if len(exceedances) < 3:
                return {"threshold_u": float(u), "sigma_scale": 0.0, "xi_shape": 0.0, "count_above_u": len(exceedances)}
        
            xi, _, sigma = genpareto.fit(exceedances, floc=0)
            
            return {
                "threshold_u": float(u),
                "sigma_scale": float(sigma),
                "xi_shape": float(xi),
                "count_above_u": len(exceedances)
            }
        
        def calculate_aml_risk_metrics(params, confidence_level, mode='high'):
            if params["sigma_scale"] == 0:
                return {"VaR": 0.0, "ES": 0.0}
        
            shape = params["xi_shape"]
            scale = params["sigma_scale"]
            u = params["threshold_u"]
            
            # Calculate VaR in the distribution space
            var_raw = genpareto.ppf(confidence_level, shape, loc=u, scale=scale)
        
            # Calculate Expected Shortfall
            if shape < 1:
                es_raw = (var_raw / (1 - shape)) + (scale - (shape * u)) / (1 - shape)
            else:
                es_raw = var_raw 
        
            if mode == 'low':
                # Re-invert back to actual dollar amounts for the micro-tail
                # Since 1/amount was used, the high 'extreme' value corresponds to a tiny dollar amount
                var_final = round(float(1 / var_raw), 4)
                es_final = round(float(1 / es_raw), 4)
            else:
                var_final = round(float(var_raw), 2)
                es_final = round(float(es_raw), 2)
        
            return {
                f"VaR_{int(confidence_level*1000)}": var_final,
                f"ES_{int(confidence_level*1000)}": es_final
            }
        
        def generate_evt_samples(params, n, mode, sample_low, sample_high):
            if params["sigma_scale"] <= 0:
                return []
            
            u_samples = np.random.uniform(sample_low, sample_high, n)
            raw_outliers = genpareto.ppf(u_samples, params["xi_shape"], loc=0, scale=params["sigma_scale"])
            
            if mode == 'high':
                return np.round(params["threshold_u"] + raw_outliers, 2).tolist()
            else:
                # Re-invert back to original dollar amounts
                inverted_values = params["threshold_u"] + raw_outliers
                return np.round(1 / inverted_values, 4).tolist()
        
        # --- 2. MAIN EXECUTION PIPELINE ---
        
        def run_evt_calibration(tx_window, scenario_type, col_amount=COL_AMOUNT, config=None):
            if config is None:
                
                # These variables are now pulled directly from the top-level arguments
                config = {
                    "threshold_pct": threshold_pct,          
                    "confidence_level": confidence_level,    
                    "sample_n": sample_n,                
                    "sample_range": sample_range   
                }
            
            # 1. Calibrate Tail Typologies 
            # High Tail
            high_params = calibrate_gpd_params(tx_window[col_amount], config["threshold_pct"], mode='high')
            high_risk = calculate_aml_risk_metrics(high_params, config["confidence_level"], mode='high')
            
            # Low Tail (Now including specific risk metrics)
            low_params = calibrate_gpd_params(tx_window[col_amount], config["threshold_pct"], mode='low')
            low_risk = calculate_aml_risk_metrics(low_params, config["confidence_level"], mode='low')
            
            # 2. Build the nested Structuring Payload with Dual-Tail Risk Architecture
            evt_payload = {
                scenario_type: {
                    "high_tail_outliers": {
                        "params": high_params,
                        "risk_metrics": high_risk,
                        "synthetic_samples": generate_evt_samples(
                            high_params, 
                            config["sample_n"], 
                            'high', 
                            config["sample_range"][0], 
                            config["sample_range"][1]
                        )
                    },
                    "low_tail_outliers": {
                        "params": low_params,
                        "risk_metrics": low_risk,
                        "synthetic_samples": generate_evt_samples(
                            low_params, 
                            config["sample_n"], 
                            'low', 
                            config["sample_range"][0], 
                            config["sample_range"][1]
                        )
                    },
                    "metadata": {
                        "model": "Generalized Pareto Distribution (GPD)",
                        "population_count": len(tx_window),
                        "scenario_tag": scenario_type,
                        "applied_config": config  
                    }
                }
            }
            
            filename = f"evt_{scenario_type}.json"
            with open(filename, "w") as f:
                json.dump(evt_payload, f, indent=4)
            
            return evt_payload
    
    
        # CRITICAL ADDITION: The outer function must return the execution of the inner function
        return run_evt_calibration(tx_window, scenario_type)

    
    # --- Midstream: Load scenario_config and restore lambdas ---

    #if not globals().get("_SCENARIO_CONFIG_DEFINED", False):
    import json

    # --- Load JSON ---
    with open("scenario_config.json", "r") as f:
        scenario_config = json.load(f)


        scenario_config["structuring"]["offshore_pool"] = _HIGH_RISK + ["BVI","CAY","PANAMA"]
        scenario_config["structuring"]["domestic_pool"] = _ASIA + _MAJOR
        scenario_config["structuring"]["country_pool_primary"] = _HIGH_RISK + _MAJOR + _ASIA
        
        scenario_config["velocity_spike"]["country_pool_primary"] = _HIGH_RISK + _ASIA + _MAJOR + ["LOCAL","SG","HK"]
        scenario_config["layering"]["country_pool_primary"] = _HIGH_RISK + _ASIA + _MAJOR + ["LOCAL"]

        scenario_config["round_trip"]["offshore_pool"] = _HIGH_RISK + ["BVI","CAY","PANAMA"]
        scenario_config["round_trip"]["domestic_pool"] = _ASIA + _MAJOR
        scenario_config["round_trip"]["country_pool_primary"] = _HIGH_RISK + _MAJOR + _ASIA

    
    # -----------------------------------------------------------
    # Remove all lambdas: convert every demographic factor to 1
    # -----------------------------------------------------------
    def neutralize_demographic_factors(cfg_section):
        """
        Replace all demographic_factors expressions with constant 1.
        This guarantees downstream code NEVER calls a lambda.
        """
        if "demographic_factors" in cfg_section:
            for k in cfg_section["demographic_factors"].keys():
                cfg_section["demographic_factors"][k] = 1  # constant multiplier

        # Corporate sub-section (for personal_to_corp)
        if "corporate" in cfg_section:
            for k in cfg_section["corporate"].keys():
                cfg_section["corporate"][k] = 1

    # -----------------------------------------------------------
    # Apply neutralization to ALL scenarios
    # -----------------------------------------------------------
    #for scenario_name, cfg in scenario_config.items():
        #neutralize_demographic_factors(cfg)

    # -----------------------------------------------------------
    # Set flag
    # -----------------------------------------------------------
    #_SCENARIO_CONFIG_DEFINED = True
    
    # --- Safe access ---
    config = scenario_config.get(scenario_type, scenario_config["structuring"])
    
    # --- Update global ---
    #global SCENARIO_CONFIG
    #if "SCENARIO_CONFIG" not in globals():
        #SCENARIO_CONFIG = {}
    #SCENARIO_CONFIG.update(scenario_config)


    #config = scenario_config.get(scenario_type, scenario_config['structuring'])
    
    # ------------------ Scenario Logic ------------------


    if scenario_type == "structuring":
        cfg = scenario_config['structuring']
        df_cfg = cfg['demographic_factors']

        # -------------------------------
        # 1. Sample synthetic demographics / entity features
        # -------------------------------
        #actor = {
            #"PEP": np.random.binomial(1,  cfg['demographics']['PEP_prob']),
            #"NatRisk": np.random.choice([0,1,2], p= cfg['demographics']['NatRisk_probs']),
            #"AgeGroup": np.random.choice([0,1,2], p= cfg['demographics']['AgeGroup_probs']),
            #"Occ": np.random.choice([0,1,2,3], p= cfg['demographics']['Occ_probs']),
            #"Income": np.random.choice([0,1,2], p= cfg['demographics']['Income_probs']),
            #"EntityType": np.random.choice([0,1,2], p= cfg['demographics']['EntityType_probs']),
            #"Industry": np.random.choice([0,1,2,3], p= cfg['demographics']['Industry_probs']),

            # --- NEW ADDITIONS FOR 2026 CALIBRATION ---
            #"Channel": np.random.choice([0,1,2,3], p=cfg['channel_probs']),
            #"Network": np.random.choice([0,1,2], p=cfg['network_probs']),
            #"Jurisdiction": np.random.choice([0,1,2], p=cfg['jurisdiction_probs'])
            # ------------------------------------------
            
        #}

        # Helper to fetch and sample dynamically
        def dynamic_sample(field_name, sub_path=None):
            """
            Dynamically selects an index based on the probability vector length.
            """
            prob_key = f"{scenario_type}_{field_name}_probs"
            
            # 1. Resolve path (Demographics are nested; Calibration is flat)
            #default_probs = cfg.get(sub_path, {}).get(prob_key) if sub_path else cfg.get(prob_key)

            default_probs = cfg.get(sub_path, cfg).get(f"{field_name}{'_prob' if field_name == 'PEP' else '_probs'}")

            
            # 2. Prioritize Bayesian shifts from Optuna overrides
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            # 3. Dynamic Range selection: np.random.choice(int) creates range(int)
            # This prevents hardcoding like [0,1,2,3] and matches p_vector 1:1
            return np.random.choice(len(p_vector), p=p_vector)


        # --- RECONCILED PROBABILITY LOOKUPS ---
        # 1. We check if Optuna has produced a warped distribution (e.g., 'NatRisk_probs')
        # 2. We fall back to the prior in scenario_config if no override exists
       
        # Execution in generate_events_from_params
        actor = {
            #"PEP": np.random.binomial(1, opt_overrides.get('PEP_prob', cfg['demographics']['PEP_prob'])),
            "PEP": np.random.binomial(1, opt_overrides.get(f"{scenario_type}_PEP_prob", cfg['demographics']['PEP_prob']))
            
            # Demographics (Nested in cfg['demographics'])
            "NatRisk":      dynamic_sample("NatRisk", "demographics"),      
            "AgeGroup":     dynamic_sample("AgeGroup", "demographics"),     
            "Occ":          dynamic_sample("Occ", "demographics"),          
            "Income":       dynamic_sample("Income", "demographics"),       
            "EntityType":   dynamic_sample("EntityType", "demographics"),   
            "Industry":     dynamic_sample("Industry", "demographics"),     
        
            # Calibration (Flat in cfg)
            "Channel":      dynamic_sample("channel"),                      
            "Network":      dynamic_sample("network"),                      
            "Jurisdiction": dynamic_sample("jurisdiction")                  
        }


        # Helper function to handle flattened dictionary from Bayesian Optimization
        def get_factor(source, actor_val):
            # If it's the expected dict, do a lookup
            if isinstance(source, dict):
                return source.get(str(actor_val), 1.0)
            # If it's a flattened number (like your debug showed), use it directly
            if isinstance(source, (int, float)):
                return float(source)
            return 1.0
        
        # 2. Compound Scaling (Active Lookup)
        # Bayesian Optimization tunes these weights to create "detectable clusters"

        # --- 2. Safe Lookups (Handles both Dicts and Ints) ---
        #PEP_f      = get_factor(df_cfg.get('PEP'), actor['PEP'])
        #NatRisk_f  = get_factor(df_cfg.get('NatRisk'), actor['NatRisk'])
        #Occ_f      = get_factor(df_cfg.get('Occ'), actor['Occ'])
        #Income_f   = get_factor(df_cfg.get('Income'), actor['Income'])
        #Entity_f   = get_factor(df_cfg.get('EntityType'), actor['EntityType'])
        #Industry_f = get_factor(df_cfg.get('Industry'), actor['Industry'])

        # --- NEW LOOKUPS ---
        #Channel_f  = get_factor(df_cfg.get('Channel'), actor['Channel'])
        #Network_f  = get_factor(df_cfg.get('Network'), actor['Network'])
        #Juris_f    = get_factor(df_cfg.get('Jurisdiction'), actor['Jurisdiction'])        
        # -------------------

        # 1. Calculate the raw Product (allowing BO to explore high values)        
        # The Total_Risk_Multiplier provides the "Separability Gap" needed for Precision
        #Total_Risk = (PEP_f * NatRisk_f * Occ_f * Income_f * Entity_f * Industry_f * Channel_f * Network_f * Juris_f )


        # --- RECONCILED LOOKUPS ---
        # 1. We check opt_overrides for a Bayesian suggestion
        # 2. We fall back to the default get_factor(df_cfg...) if no suggestion exists
        # Final 2026 Reconciled Lookups using local scenario_type
        PEP_f      = opt_overrides.get(f"{scenario_type}_PEP_f", get_factor(df_cfg.get('PEP'), actor['PEP']))
        NatRisk_f  = opt_overrides.get(f"{scenario_type}_NatRisk_f", get_factor(df_cfg.get('NatRisk'), actor['NatRisk']))
        Occ_f      = opt_overrides.get(f"{scenario_type}_Occ_f", get_factor(df_cfg.get('Occ'), actor['Occ']))
        Income_f   = opt_overrides.get(f"{scenario_type}_Income_f", get_factor(df_cfg.get('Income'), actor['Income']))
        Entity_f   = opt_overrides.get(f"{scenario_type}_Entity_f", get_factor(df_cfg.get('EntityType'), actor['EntityType']))
        Industry_f = opt_overrides.get(f"{scenario_type}_Industry_f", get_factor(df_cfg.get('Industry'), actor['Industry']))
        
        Channel_f  = opt_overrides.get(f"{scenario_type}_Channel_f", get_factor(df_cfg.get('Channel'), actor['Channel']))
        Network_f  = opt_overrides.get(f"{scenario_type}_Network_f", get_factor(df_cfg.get('Network'), actor['Network']))
        Juris_f    = opt_overrides.get(f"{scenario_type}_Juris_f", get_factor(df_cfg.get('Jurisdiction'), actor['Jurisdiction']))
        
        
        # Final Product
        Total_Risk = (PEP_f * NatRisk_f * Occ_f * Income_f * Entity_f * Industry_f * Channel_f * Network_f * Juris_f)



        
        # --- IMPROVED CALIBRATION STEP (2026 RISK-WEIGHTED) ---
        
        # Higher risk profiles (Total_Risk) are more likely to exhibit aggressive structuring
        # We use a Sigmoid-like function or a risk-scaled probability
        #aggression_prob = min(0.9, 0.15 + (Total_Risk / 50.0)) # Scaled by your risk factors

        #if np.random.random() < aggression_prob:
            # Aggressive intensity is also scaled by the risk profile
            # Ensures that 'High-Risk Network' actors are significantly more aggressive
            #intensity = np.random.uniform(1.6, 2.5) * (1 + (Network_f - 1) * 0.2)
        #else:
            #intensity = 1.0

        # We flag if intensity is high AND behavior is coupled with high-risk subsegments
        #if (intensity > 1.5 and n_credits > cfg['n_credits_base'] * 1.2) or \
           #(intensity > 1.2 and (Network_f > 2.0 or Juris_f > 3.0)):
            #current_label = 1 # Truly Suspicious: Strategic structuring across risks
        #else:
            #current_label = 0 # High-risk profile, but behavior remains "Latent" or within norms
            

        # --- 2026 HOLISTIC INTENSITY REFACTOR ---
        
        # High Total_Risk increases the probability of aggressive behavior
        aggression_prob = min(0.9, 0.15 + (Total_Risk / 50.0))


        

        if np.random.random() < aggression_prob:
            # INTEGRATED LOGIC: Intensity is now proportional to Total_Risk
            # We use a base intensity (1.6 - 2.5) further amplified by the risk profile
            # This ensures demographic, channel, and network risks all scale the 'burst'
            risk_scaling = 1 + (Total_Risk / 100.0)
            intensity = np.random.uniform(1.6, 2.5) * risk_scaling
        else:
            # Baseline intensity remains near 1.0 to protect the 'Normal' class
            intensity = np.random.uniform(0.9, 1.1)



        # --- 1. THE 2026 UBR UPGRADE (Quick Fix) ---
        # Safeguard the product of your 9 multipliers against BO-driven zeros
        Total_Behavioral_Risk = max(1e-6, Total_Risk)

        # --- 2. DYNAMIC INTENSITY ENGINE ---
        # Establish the Mode: 25% chance of suspicious "Burst" structuring
        # We use a base intensity that BO can then scale via the UBR
        if np.random.random() < 0.25:
            base_intensity = np.random.uniform(1.6, 2.5)
            intensity = base_intensity * Total_Behavioral_Risk
        else:
            # Baseline: Normal users also scaled by risk, but at a lower plateau
            intensity = np.random.uniform(0.9, 1.1) * (Total_Behavioral_Risk * 0.5)

        # --- 3. BAYESIAN GOVERNOR (Crucial for F1) ---
        # Replace the hard-coded scaling with BO-suggested boundaries
        # This prevents "Mathematical Explosion" while keeping the signal sharp
        str_int_floor = trial.suggest_float("str_int_floor", 0.1, 1.0)
        str_int_ceiling = trial.suggest_float("str_int_ceiling", 4.0, 12.0)

        intensity = max(str_int_floor, min(str_int_ceiling, round(intensity, 2)))



        # --- 2. DYNAMIC INTENSITY ENGINE (Reconciled) ---
        # Parameters to control the "Suspicious" (High) vs "Normal" (Low) intensity logic
        # 'mode_split' replaces the hardcoded 0.25 chance
        mode_split   = opt_overrides.get(f"{scenario_type}_mode_split", 0.25)
        high_int_min = opt_overrides.get(f"{scenario_type}_high_int_min", 1.6)
        high_int_max = opt_overrides.get(f"{scenario_type}_high_int_max", 2.5)
        low_int_min  = opt_overrides.get(f"{scenario_type}_low_int_min", 0.9)
        low_int_max  = opt_overrides.get(f"{scenario_type}_low_int_max", 1.1)
        low_int_scale = opt_overrides.get(f"{scenario_type}_low_int_scale", 0.5)

        if np.random.random() < mode_split:
            base_intensity = np.random.uniform(high_int_min, high_int_max)
            intensity = base_intensity * Total_Behavioral_Risk
        else:
            # Baseline: Normal users also scaled by risk, but at a lower plateau
            intensity = np.random.uniform(low_int_min, low_int_max) * (Total_Behavioral_Risk * low_int_scale)

        # --- 3. BAYESIAN GOVERNOR ---
        # Keeps boundaries dynamic as per your script's requirement
        # --- 3. BAYESIAN GOVERNOR (Sourced from Overrides) ---
        # No Priors enqueued; Optuna explores these ranges starting from trial 1
        str_int_floor   = opt_overrides.get(f"{scenario_type}_str_int_floor", 0.1)
        str_int_ceiling = opt_overrides.get(f"{scenario_type}_str_int_ceiling", 12.0)

        intensity = max(str_int_floor, min(str_int_ceiling, round(intensity, 2))


        
        # --- IMPROVED LABELING LOGIC ---
        
        # Suspicion is now a combination of volume, velocity, and multi-channel risk
        risk_threshold = 1.8 # Baseline for 'Grey Zone' entry
        
        # Labels are now grounded in the holistic 'Total_Risk' score
        if (intensity > 1.4 and Total_Risk > risk_threshold) or \
           (intensity > 1.2 and (Network_f > 2.0 or Juris_f > 2.5)):
            current_label = 1
        else:
            current_label = 0



        # --- 1. DYNAMIC DETECTION THRESHOLD (2026 Standard) ---
        # Instead of a static 1.8, we use a sliding scale. 
        # Higher UBR = a lower 'Intensity' requirement to be labeled suspicious.
        # This allows BO to find the "Grey Zone" where criminals try to blend in.
        dynamic_threshold = max(1.1, 2.5 - (Total_Behavioral_Risk * 0.15))

        # --- 2. MULTI-PRONGED DETECTION TRIGGERS ---
        
        # Trigger A: Profile-Adjusted Velocity Surge
        is_velocity_spike = (intensity > dynamic_threshold)

        # Trigger B: Obfuscation-Driven Risk (2026 Red Flag)
        # If the Network or Jurisdiction risk is high, even moderate intensity is flagged.
        is_obfuscation_trigger = (intensity > 1.2 and (Network_f > 2.0 or Juris_f > 2.2))

        # Trigger C: High-Intensity Shadow Rail (Channel Risk)
        # Using specific channels (like Stablecoins or Instant Rails) at high speed
        is_channel_red_flag = (intensity > 1.5 and Channel_f > 1.8)

        # --- 3. FINAL LABEL ASSIGNMENT ---
        if is_velocity_spike or is_obfuscation_trigger or is_channel_red_flag:
            current_label = 1
        else:
            current_label = 0





        # --- 1. DYNAMIC DETECTION THRESHOLD (Reconciled) ---
        # Parameters to tune the sliding scale: dynamic_threshold = max(floor, intercept - (UBR * slope))
        dyn_floor     = opt_overrides.get(f"{scenario_type}_dyn_floor", 1.1)
        dyn_intercept = opt_overrides.get(f"{scenario_type}_dyn_intercept", 2.5)
        dyn_slope     = opt_overrides.get(f"{scenario_type}_dyn_slope", 0.15)
        
        dynamic_threshold = max(dyn_floor, dyn_intercept - (Total_Behavioral_Risk * dyn_slope))

        # --- 2. MULTI-PRONGED DETECTION TRIGGERS (Reconciled) ---
        
        # Trigger A: Profile-Adjusted Velocity Surge
        is_velocity_spike = (intensity > dynamic_threshold)

        # Trigger B: Obfuscation-Driven Risk
        # Tuning the sensitivity for Network and Jurisdiction red flags
        obf_int_min    = opt_overrides.get(f"{scenario_type}_obf_int_min", 1.2)
        obf_net_limit  = opt_overrides.get(f"{scenario_type}_obf_net_limit", 2.0)
        obf_juris_limit = opt_overrides.get(f"{scenario_type}_obf_juris_limit", 2.2)
        
        is_obfuscation_trigger = (intensity > obf_int_min and 
                                  (Network_f > obf_net_limit or Juris_f > obf_juris_limit))

        # Trigger C: High-Intensity Shadow Rail (Channel Risk)
        # Tuning the specific weight of Channel risk
        chan_int_min = opt_overrides.get(f"{scenario_type}_chan_int_min", 1.5)
        chan_f_limit = opt_overrides.get(f"{scenario_type}_chan_f_limit", 1.8)
        
        is_channel_red_flag = (intensity > chan_int_min and Channel_f > chan_f_limit)

        # --- 3. FINAL LABEL ASSIGNMENT ---
        if is_velocity_spike or is_obfuscation_trigger or is_channel_red_flag:
            current_label = 1
        else:
            current_label = 0






        # 2. LOG-COMPRESSION (The 2026 RegTech Standard)
        # Instead of a hard cap, we use log1p to compress the 'Total_Risk' 
        # This preserves the 'Ordering' (higher risk still means more rows) 
        # but prevents the multiplicative explosion.
        compressed_risk = 1 + np.log1p(Total_Risk)
        
        # 3. FINAL CALCULATION
        # BO still controls n_credits_base and intensity, but the risk doesn't explode.
        n_credits = max(
            cfg['n_credits_min'],
            int(cfg['n_credits_base'] * intensity * compressed_risk)
        )

        
        
        #n_credits = max(
            #cfg['n_credits_min'],
            #int(cfg['n_credits_base'] * intensity * Total_Risk)
        #)





        # --- 2026 PRODUCTION REFINEMENT ---
        # We avoid log1p compression to preserve the 'Exponential Signature' of illicit flows.
        # Total_Behavioral_Risk now provides the authentic scaling for transaction volume.
        
        # 1. Unified Risk Scaling (Linear, not Logarithmic)
        # Replacing Total_Risk with Total_Behavioral_Risk
        risk_scaling_factor = Total_Behavioral_Risk

        # 2. FINAL VOLUME CALCULATION
        # n_credits is the 'Haystack' size for this specific actor.
        # BO still controls n_credits_base and intensity, providing the 'Separability Gap'.
        n_credits = max(
            cfg['n_credits_min'],
            int(cfg['n_credits_base'] * intensity * risk_scaling_factor)
        )


        

        
        # 3. Generate Deposits with Velocity Sensitivity
        #t_prev = anchor + timedelta(hours=Industry_f)

        # --- 2026 REFINED STARTING OFFSET ---
        # We want a random start delay that is SHORTER for high-intensity/high-risk actors
        # Base delay (e.g., 24 hours) divided by (Intensity * Risk)
        base_delay_hours = 24.0 
        #start_offset = np.random.uniform(0, base_delay_hours) / (intensity * max(0.5, Total_Risk))
        
        # 2026 Refinement: Total_Behavioral_Risk now drives the "Urgency" of the first transaction
        start_offset = np.random.uniform(0, base_delay_hours) / (intensity * max(0.5, Total_Behavioral_Risk))

        t_prev = anchor + timedelta(hours=start_offset)
        
        vs = df_cfg.get('velocity_sensitivity', 0.1)
        vos = df_cfg.get('volume_sensitivity', 0.1)


        # --- 1. REFINED STARTING OFFSET (Reconciled) ---
        base_delay_hours  = opt_overrides.get(f"{scenario_type}_base_delay_hours", 24.0)
        urgency_risk_floor = opt_overrides.get(f"{scenario_type}_urgency_risk_floor", 0.5)

        # Urgency is driven by the interaction of Intensity and Behavioral Risk
        start_offset = np.random.uniform(0, base_delay_hours) / (intensity * max(urgency_risk_floor, Total_Behavioral_Risk))

        t_prev = anchor + timedelta(hours=start_offset)


        # --- 2. SENSITIVITY SOURCE (With Fallbacks) ---
        vs = opt_overrides.get(f"{scenario_type}_velocity_sens", df_cfg.get('velocity_sensitivity', 0.1))
        vos = opt_overrides.get(f"{scenario_type}_volume_sens", df_cfg.get('volume_sensitivity', 0.1))


        # --- EVT RETROFIT: GPD-Driven Extreme Value Sampling ---
        
        # Now our call inside def generate_events_from_params will run perfectly:
        evt_payload = extreme_value_theory(
            tx_window, 
            scenario_type="structuring",
            threshold_pct=trial.suggest_float("str_threshold_pct", 90, 99),
            confidence_level=trial.suggest_float("str_confidence_level", 0.99, 0.9999),
            sample_n=trial.suggest_int("str_sample_n", 3, 15)
        )
        
        # 1. Access the EVT Params from the pre-computed payload
        # Assuming 'evt_payload' was populated via run_evt_calibration earlier
        struct_data = evt_payload["structuring"]["high_tail_outliers"]
        evt_params = struct_data["params"]


        # --- 3. EVT RETROFIT (Unknowns - No Enqueue) ---
        # Sourcing from opt_overrides populated by the upstream objective suggestions
        str_threshold_pct   = opt_overrides.get(f"{scenario_type}_str_threshold_pct", 95.0)
        str_confidence_level = opt_overrides.get(f"{scenario_type}_str_confidence_level", 0.999)
        str_sample_n        = int(opt_overrides.get(f"{scenario_type}_str_sample_n", 5))

        evt_payload = extreme_value_theory(
            tx_window, 
            scenario_type="structuring",
            threshold_pct=str_threshold_pct,
            confidence_level=str_confidence_level,
            sample_n=str_sample_n
        )

        # Extract EVT parameters from payload
        struct_data = evt_payload["structuring"]["high_tail_outliers"]
        evt_params = struct_data["params"]


        # --- OUTSIDE THE LOOP (Bayesian Suggestions) ---
        # These are the "Unknowns" the Bayesian Optimizer will solve for

        # BAYESIAN SUGGESTIONS: Define the behavioral strategy for THIS specific entity/window
        # We suggest these ONCE so the entity behaves consistently across all n_credits        
        
        mean_translation_factor = trial.suggest_float("str_mean_shift", 1.0, 1.5)  # Multiplier to push mean up
        sigma_contraction_factor = trial.suggest_float("str_sigma_scale", 0.5, 1.0) # Multiplier to tighten spread        

        # --- 4. BEHAVIORAL STRATEGY (Bayesian Suggestions) ---
        # These determine how "tight" or "shifted" the outlier distribution becomes
        mean_translation_factor  = opt_overrides.get(f"{scenario_type}_str_mean_shift", 1.2)
        sigma_contraction_factor = opt_overrides.get(f"{scenario_type}_str_sigma_scale", 0.8)


        # -------------------------------
        # 4. Generate deposits
        # -------------------------------

        for _ in range(n_credits):
            # Velocity: Time gaps compress as risk increases, creating "burst" patterns
            #gap_scale = cfg['gap_mean'] / max(0.2, (1 + vs * Total_Risk))
            #gap_hours = float(np.random.lognormal(mean=gap_scale, sigma=cfg['gap_sigma']))
            #t_trade = t_prev + timedelta(hours=gap_hours)

            # 2026 REFINEMENT: Use intensity as the primary compressor
            # Intensity (>1.5 for Cat 3) should drive the 'Criminal Velocity'
            # vs * Total_Risk provides the 'Persona-based' urgency

            #compression_factor = (intensity * (1 + vs * Total_Risk))
            
            #gap_scale = cfg['gap_mean'] / max(0.1, compression_factor)            

            # 1. TEMPORAL BRIDGE: Use Total_Behavioral_Risk and Intensity for 'The Crunch'
            # (Replaces velocity_sensitivity and Total_Risk)

            # 2026 Superior Approach: Linear Scaling for Maximum Signal Sharpness
            # We remove the log-compression and use the Unified Risk Bridge directly

            
            holistic_velocity_driver = intensity * Total_Behavioral_Risk

            velocity_floor = opt_overrides.get(f"{scenario_type}_velocity_floor", 0.1)
            risk_weight    = opt_overrides.get(f"{scenario_type}_holistic_risk_weight", 1.0)

            holistic_velocity_driver = intensity * (Total_Behavioral_Risk * risk_weight)
            
            # Final gap calculation: The 'driver' creates the temporal crunch
            gap_scale = cfg['gap_mean'] / max(0.1, holistic_velocity_driver)

            gap_scale = cfg['gap_mean'] / max(velocity_floor, holistic_velocity_driver)


            
            # Generate gap with Log-Normal variance
            gap_hours = float(np.random.lognormal(mean=np.log(gap_scale), sigma=cfg['gap_sigma']))
            
            t_trade = t_prev + timedelta(hours=gap_hours)

            
            # Volume: Mean amounts scale close to but below reporting thresholds
            #amt_mean_adj = cfg['amt_mu'] * (1 + vos * (intensity - 1)) * Income_f * PEP_f
            #amt_sigma_adj = cfg['amt_sigma'] / max(0.5, Total_Risk) # Tighter sigma for high risk

            # 2. BASE VALUE CALCULATION: Anchored to Intensity
            # (Replaces Income_f and PEP_f as they are already in Total_Behavioral_Risk)
            amt_mean_adj = cfg['amt_mu'] * (1 + (vos * (intensity - 1)))
            amt_sigma_adj = cfg['amt_sigma'] / max(0.5, Total_Behavioral_Risk)

            # 2. AMOUNT BASELINE
            # Sourcing floor for max(0.5, ...)
            risk_sigma_floor = opt_overrides.get(f"{scenario_type}_risk_sigma_floor", 0.5)
            amt_sigma_adj = cfg['amt_sigma'] / max(risk_sigma_floor, Total_Behavioral_Risk)
            
            
            # 2026 OVERHAUL: Use Truncated Normal to avoid clipping artifacts
            lower_bound = cfg['amt_min']
            upper_bound = cfg['amt_max']

            # Calculate truncation bounds (a, b) in terms of standard deviations from the mean
            a_bound = (lower_bound - amt_mean_adj) / amt_sigma_adj
            b_bound = (upper_bound - amt_mean_adj) / amt_sigma_adj

            # Generate the amount using a Truncated Normal distribution
            #amt = round(truncnorm.rvs(a_bound, b_bound, loc=amt_mean_adj, scale=amt_sigma_adj), 2)

            # 2. EVT Sampling Logic: Only trigger if we have a valid GPD fit and high intensity
            #if evt_params["sigma_scale"] > 0 and intensity > 1.2:
                # Use Total_Risk to expand the 'scale' (the spread of the extreme tail)
                #dynamic_sigma = evt_params["sigma_scale"] * (1 + (vos * Total_Risk))
                
                # Determine tail depth: high compressed_risk pushes sampling to higher percentiles
                #tail_depth = min(0.99, 0.1 + (compressed_risk / 10.0))
                #High-risk entities sample from the upper 50% to 'tail_depth' of the GPD
                #u_sample = np.random.uniform(0.5, tail_depth)

            # 3. EVT SAMPLING (THE HIGH TAIL)
            evt_cutoff = opt_overrides.get(f"{scenario_type}_evt_intensity_cutoff", 1.4)
            
            # 3. HYBRID AMOUNT LOGIC: EVT Shocks vs. Bayesian Facade
            #if evt_params["sigma_scale"] > 0 and intensity > 1.4:

            if evt_params["sigma_scale"] > 0 and intensity > evt_cutoff:
                
                # 2026 REFINEMENT: Total_Behavioral_Risk expands the 'Fat Tail'
                dynamic_sigma = evt_params["sigma_scale"] * (1 + (vos * Total_Behavioral_Risk))
                # Tail depth linked to UBR to find the 'Needle' in 2026 space
                tail_depth = min(0.995, 0.5 + (Total_Behavioral_Risk / 20.0))

                # Sourcing tail depth math: min(0.995, 0.5 + Risk/20.0)
                td_base = opt_overrides.get(f"{scenario_type}_tail_depth_base", 0.5)
                td_div  = opt_overrides.get(f"{scenario_type}_tail_depth_div", 20.0)
                tail_depth = min(0.995, td_base + (Total_Behavioral_Risk / td_div))

                
                u_sample = np.random.uniform(0.6, tail_depth)

                # Sourcing uniform sample low: np.random.uniform(0.6, ...)
                ts_low = opt_overrides.get(f"{scenario_type}_tail_sample_low", 0.6)
                u_sample = np.random.uniform(ts_low, tail_depth)
                
                
                # Generate the extreme exceedance over the threshold
                exceedance = genpareto.ppf(u_sample, evt_params["xi_shape"], loc=0, scale=dynamic_sigma)
                
                # Final amount = Threshold + Extreme Deviation (Capped at amt_max)
                #amt = round(min(cfg['amt_max'], evt_params["threshold_u"] + exceedance), 2)
                amt = round(evt_params["threshold_u"] + exceedance, 2)

                

                
            else:
                # Low-Risk: Fallback to a standard baseline or the threshold itself
                # This prevents "normal" behavior from accidentally triggering extreme EVT logic
                #amt = round(truncnorm.rvs(a_bound, b_bound, loc=amt_mean_adj, scale=amt_sigma_adj), 2)
    
                # 1. Apply Bayesian Mean Translation
                # We multiply the original adjusted mean by the translation factor
                # This pushes the "bulk" of transactions closer to the threshold

                # 4. NORMAL DISTRIBUTION (TRUNCATED)
                # Sourcing tuned_sigma floor: max(0.01, ...)
                sigma_min = opt_overrides.get(f"{scenario_type}_tuned_sigma_min", 0.01)

                
                tuned_mean = amt_mean_adj * mean_translation_factor
                
                # 2. Apply Bayesian Sigma Scaling
                # We multiply the sigma; values < 1.0 make the behavior more "precise" and predictable
                tuned_sigma = amt_sigma_adj * sigma_contraction_factor
                
                # 3. Recalculate Truncation Bounds based on the NEW tuned parameters
                # This is critical so the distribution shape stays valid within [min, max]
                a_tuned = (lower_bound - tuned_mean) / max(0.01, tuned_sigma)
                b_tuned = (upper_bound - tuned_mean) / max(0.01, tuned_sigma)

                a_tuned = (lower_bound - tuned_mean) / max(sigma_min, tuned_sigma)
                b_tuned = (upper_bound - tuned_mean) / max(sigma_min, tuned_sigma)                
                
                # 4. Generate the "Sophisticated" Baseline Amount
                amt = round(truncnorm.rvs(a_tuned, b_tuned, loc=tuned_mean, scale=tuned_sigma), 2)                


            
            
            # --- 2026 DIRECTIONAL OVERHAUL ---
            # Structuring is 90% Inbound (Placement), but 10% Outbound (Layering/Withdrawal)
            # This creates "Hard Negatives" that boost Precision.
            #tx_type = np.random.choice(["CREDIT", "DEBIT"], p=[0.9, 0.1])


            # --- 2026 HYBRID NAMING OVERHAUL ---
            # 1. Define the Direction (Placement bias)
            direction = np.random.choice(["CREDIT", "DEBIT"], p=[0.9, 0.1])

            # 5. DIRECTION & CHANNEL
            # Sourcing transaction probabilities: p=[0.9, 0.1]
            c_p = opt_overrides.get(f"{scenario_type}_credit_p", 0.9)
            d_p = opt_overrides.get(f"{scenario_type}_debit_p", 0.1)
            direction = np.random.choice(["CREDIT", "DEBIT"], p=[c_p, d_p])

            
            # Select the label based on the probabilities in the config
            idx = np.random.choice(len(cfg['channel_probs']), p=cfg['channel_probs'])
            current_chan = cfg['channel_labels'][idx] # Programmatic access!
            
            # This label then becomes part of our high-cardinality tx_type
            # which TME uses to boost your F1 score to 0.80+
            tx_type = f"{direction}_{current_chan}"

            
            #rows.append(_assemble_tx(acct, party_key, ref_date, t_trade, amt, "CREDIT", config, label=current_label))            #rows.extend(_generate_micro_tx(t_trade, amt, config, ref_date=ref_date, label=current_label))
            rows.append(_assemble_tx(acct, party_key, ref_date, t_trade, amt, tx_type, config, label=current_label))
            
            t_prev = t_trade
            
        # --- MICRO-SMURFING (High-Velocity Signal) ---
        if current_label == 1:
            # Suspicious actors often 'test' the system with tiny micro-credits/debits
            # Use your config's micro_tx_types: ["DEBIT", "CREDIT"]
            rows.extend(_generate_micro_tx(t_trade, amt, config, ref_date=ref_date, label=current_label))

        #rows.extend(_generate_micro_tx(t_trade, amt, config, ref_date=ref_date, label=current_label))



    
    
    # ------------------ Velocity Spike Function ------------------

    elif scenario_type == "velocity_spike":
        cfg = scenario_config['velocity_spike']
        # The "Handshake": Pull weights directly from the optimized JSON branch
        dw = cfg["demographics"] 
        
        # 1. Sample Actor Attributes
        is_pep = random.random() < 0.1
        is_biz = random.random() < 0.3
        is_young = random.choice([True, False])
        risk_score = random.uniform(0, 1)


        

        # --- 2026 OPTIMIZED VELOCITY ACTOR SAMPLING ---
        # Instead of fixed 0.1 or 0.3, we use Optuna overrides to warp the 
        # probability of selecting a high-risk actor profile.
        
        # 1. Sample PEP (Boolean)
        # Optuna tunes 'velocity_spike_PEP_prob' to find the best mix for detection
        pep_prob = opt_overrides.get(f"{scenario_type}_PEP_prob", 0.1)
        is_pep = np.random.binomial(1, pep_prob) == 1
        
        # 2. Sample Entity Type (Business vs Individual)
        # Optuna tunes 'velocity_spike_Business_prob'
        biz_prob = opt_overrides.get(f"{scenario_type}_Business_prob", 0.3)
        is_biz = np.random.binomial(1, biz_prob) == 1
        
        # 3. Sample Youth Status (Boolean)
        # Optuna tunes 'velocity_spike_Youth_prob' to trigger 'youth_boost' multipliers
        youth_prob = opt_overrides.get(f"{scenario_type}_Youth_prob", 0.5)
        is_young = np.random.binomial(1, youth_prob) == 1
        
        # 4. Sample Baseline Risk Score (Continuous 0.0 - 1.0)
        # Optuna can shift the alpha/beta of a Beta distribution to skew risk higher/lower
        # but for a direct replacement of random.uniform:
        risk_shift = opt_overrides.get(f"{scenario_type}_risk_skew", 1.0)
        # Power-shifting a uniform sample skews the baseline actor risk
        risk_score = np.random.uniform(0, 1) ** (1 / risk_shift)

        # Construct actor object for downstream multiplier application
        actor = {
            "PEP": is_pep,
            "EntityType": 1 if is_biz else 0, # Mapping Business to 1
            "AgeGroup": 0 if is_young else 1,  # Mapping Young to 0
            "BaseRisk": risk_score
        }

        
        # --- NEW CODE: Incorporate Network/Channel/Economic Factors into Risk Score ---
        # We need proxy variables for these new dimensions for demonstration purposes in this snippet
        # In a full simulator, these would be sampled like the other demographics.
        network_centrality_score = random.uniform(1.0, cfg['network_topology']['centrality_threshold'])
        channel_hop_score = random.uniform(1.0, cfg['channel_config']['hop_intensity_multiplier'])
        savings_drain_ratio = random.uniform(0.1, cfg['economic_sensibility']['savings_drain_threshold'])



        # 1. Resolve Dictionaries
        topo = cfg.get('network_topology', {})
        chan = cfg.get('channel_config', {})
        econ = cfg.get('economic_sensibility', {})

        # 2. Extract Overrides using your established pattern
        # This resolves: Optuna -> Config Dict -> Hardcoded Default
        c_limit = opt_overrides.get(f"{scenario_type}_centrality_threshold", topo.get('centrality_threshold', 3.5))
        h_limit = opt_overrides.get(f"{scenario_type}_hop_multiplier", chan.get('hop_intensity_multiplier', 2.2))
        d_limit = opt_overrides.get(f"{scenario_type}_drain_threshold", econ.get('savings_drain_threshold', 0.9))

        # --- OPTIMIZED BOUNDS (REPLACING HARD-CODED 1.0 AND 0.1) ---
        c_floor = opt_overrides.get(f"{scenario_type}_centrality_floor", 1.0)
        h_floor = opt_overrides.get(f"{scenario_type}_hop_floor", 1.0)
        d_floor = opt_overrides.get(f"{scenario_type}_drain_floor", 0.1)
        
        # 3. Dynamic Sampling (2026 Optimization)
        # The 'Physics' of the simulation is now controlled by the Bayesian Loop

        network_centrality_score = random.uniform(1.0, cfg['network_topology']['centrality_threshold'])
        channel_hop_score = random.uniform(1.0, cfg['channel_config']['hop_intensity_multiplier'])
        savings_drain_ratio = random.uniform(0.1, cfg['economic_sensibility']['savings_drain_threshold'])

        
        network_centrality_score = random.uniform(c_floor, c_limit)
        channel_hop_score        = random.uniform(h_floor, h_limit)
        savings_drain_ratio      = random.uniform(d_floor, d_limit)

        
        # 2. Compute COMPOUND Scaling Factors (Multiplicative for Huge Size)
        # We multiply instead of adding to ensure the "Spike" is unmistakable
        F_burst = ( (dw["F_burst"]["PEP"] if is_pep else 1.0) * 
                    (dw["F_burst"]["Business"] if is_biz else 1.0) * 
                    (1 + risk_score * dw["F_burst"]["risk_score_scale"]) )


        # --- F_burst: Calibrated Multipliers ---
        # Optuna targets: PEP impact, Business impact, and the Risk Score Scale
        f_burst_pep = opt_overrides.get(f"{scenario_type}_F_burst_PEP", dw["F_burst"].get("PEP", 2.5))
        f_burst_biz = opt_overrides.get(f"{scenario_type}_F_burst_Business", dw["F_burst"].get("Business", 1.5))
        f_burst_rss = opt_overrides.get(f"{scenario_type}_F_burst_risk_scale", dw["F_burst"].get("risk_score_scale", 2.0))

        f_burst_pep_n = opt_overrides.get(f"{scenario_type}_F_burst_PEP_neutral", 1.0)
        f_burst_biz_n = opt_overrides.get(f"{scenario_type}_F_burst_Business_neutral", 1.0)
        
        F_burst = ( (f_burst_pep if is_pep else f_burst_pep_n) * 
                    (f_burst_biz if is_biz else f_burst_biz_n) * 
                    (1 + risk_score * f_burst_rss) 


        F_size = ( (1 + risk_score * dw["F_size"]["risk_score_scale"]) * 
                   (dw["F_size"]["Business"] if is_biz else 1.0) * 
                   (dw["F_size"]["youth_boost"] if is_young else 1.0) )

        # --- F_size: Calibrated Multipliers ---
        # Optuna targets: Youth boost, Business impact, and Risk Score Scale
        f_size_rss   = opt_overrides.get(f"{scenario_type}_F_size_risk_scale", dw["F_size"].get("risk_score_scale", 1.8))
        f_size_biz   = opt_overrides.get(f"{scenario_type}_F_size_Business", dw["F_size"].get("Business", 1.4))
        f_size_youth = opt_overrides.get(f"{scenario_type}_F_size_youth_boost", dw["F_size"].get("youth_boost", 1.2))

        f_size_biz_n   = opt_overrides.get(f"{scenario_type}_F_size_Business_neutral", 1.0)
        f_size_youth_n = opt_overrides.get(f"{scenario_type}_F_size_youth_neutral", 1.0)
        
        F_size = ( (1 + risk_score * f_size_rss) * 
                   (f_size_biz if is_biz else f_size_biz_n) * 
                   (f_size_youth if is_young else f_size_youth_n) )

        
        # Compression factors for Velocity (Timing)
        F_inter = (1 + risk_score * dw["F_inter"]["risk_score_scale"]) * (dw["F_inter"]["youth_boost"] if is_young else 1.0)


        # --- F_inter: Inter-arrival Calibration ---
        f_inter_rss   = opt_overrides.get(f"{scenario_type}_F_inter_risk_scale", dw["F_inter"].get("risk_score_scale", 2.5))
        f_inter_youth = opt_overrides.get(f"{scenario_type}_F_inter_youth_boost", dw["F_inter"].get("youth_boost", 1.5))
        f_inter_y_n   = opt_overrides.get(f"{scenario_type}_F_inter_youth_neutral", 1.0)

        F_inter = (1 + risk_score * f_inter_rss) * (f_inter_youth if is_young else f_inter_y_n)

        
        F_intra = (1 + risk_score * dw["F_intra"]["risk_score_scale"]) * (dw["F_intra"]["youth_boost"] if is_young else 1.0)


        # --- F_intra: Intra-burst Calibration ---
        f_intra_rss   = opt_overrides.get(f"{scenario_type}_F_intra_risk_scale", dw["F_intra"].get("risk_score_scale", 3.0))
        f_intra_youth = opt_overrides.get(f"{scenario_type}_F_intra_youth_boost", dw["F_intra"].get("youth_boost", 2.0))
        f_intra_y_n   = opt_overrides.get(f"{scenario_type}_F_intra_youth_neutral", 1.0)

        F_intra = (1 + risk_score * f_intra_rss) * (f_intra_youth if is_young else f_intra_y_n)

        
        F_amt = ( (dw["F_amt"]["PEP"] if is_pep else 1.0) * (1 + risk_score * dw["F_amt"]["risk_score_scale"]) )

        # --- F_amt: Amount Scaling Calibration ---
        f_amt_pep   = opt_overrides.get(f"{scenario_type}_F_amt_PEP", dw["F_amt"].get("PEP", 2.2))
        f_amt_pep_n = opt_overrides.get(f"{scenario_type}_F_amt_PEP_neutral", 1.0)
        f_amt_rss   = opt_overrides.get(f"{scenario_type}_F_amt_risk_scale", dw["F_amt"].get("risk_score_scale", 1.5))

        F_amt = (f_amt_pep if is_pep else f_amt_pep_n) * (1 + risk_score * f_amt_rss)        

        # --- NEW CODE: Define F_type factors based on attributes ---
        # Business accounts tend to have more DEBIT/CREDIT (supplier payments/customer receipts)
        # Young individuals might lean slightly more towards P2P
        #F_type_debit = 1.0 + (0.3 if is_biz else 0.0)
        #F_type_credit = 1.0 + (0.3 if is_biz else 0.0)

        # 2026 Optimized Logic: Asymmetrical Scaling
        # Business spikes are usually directional (e.g., rapid dispersal)
        # We increase DEBIT more aggressively to simulate 'Layering Outflow'
        F_type_debit = 1.0 + (0.5 if is_biz else 0.0)
        F_type_credit = 1.0 + (0.2 if is_biz else 0.0)


        
        # --- F_type: Asymmetrical Scaling Calibration (Block 3) ---
        
        # Business Spikes: Layering Outflow (DEBIT > CREDIT)
        f_debit_n = opt_overrides.get(f"{scenario_type}_F_type_debit_neutral", 1.0)
        debit_boost_biz  = opt_overrides.get(f"{scenario_type}_F_type_debit_biz_boost", 0.5)
        f_debit_nonbiz = opt_overrides.get(f"{scenario_type}_F_type_debit_nonbiz_val", 0.0)
        
        F_type_debit = f_debit_n + (debit_boost_biz if is_biz else f_debit_nonbiz)

        
        f_credit_n = opt_overrides.get(f"{scenario_type}_F_type_credit_neutral", 1.0)        
        credit_boost_biz = opt_overrides.get(f"{scenario_type}_F_type_credit_biz_boost", 0.2)
        f_credit_nonbiz = opt_overrides.get(f"{scenario_type}_F_type_credit_nonbiz_val", 0.0)
        
        F_type_credit = f_credit_n + (credit_boost_biz if is_biz else f_credit_nonbiz)

        
        #F_type_p2p = 1.0 + (0.2 if is_young else 0.0)
        F_type_p2p = 1.0 + (0.2 if is_young else 0.0) + (0.5 if network_centrality_score > 2.5 else 0.0)


        # P2P Scaling: Youth and Network Centrality
        f_p2p_n = opt_overrides.get(f"{scenario_type}_F_type_p2p_neutral", 1.0)
        p2p_young_boost = opt_overrides.get(f"{scenario_type}_F_type_p2p_young_boost", 0.2)
        f_p2p_nonyouth = opt_overrides.get(f"{scenario_type}_F_type_p2p_nonyouth_val", 0.0)
        p2p_cent_boost  = opt_overrides.get(f"{scenario_type}_F_type_p2p_centrality_boost", 0.5)
        cent_p2p_thresh = opt_overrides.get(f"{scenario_type}_centrality_p2p_threshold", 2.5) # The hardcoded 2.5 limit
        f_p2p_lowcent = opt_overrides.get(f"{scenario_type}_F_type_p2p_lowcentral_val", 0.0)
        
        
        F_type_p2p = f_p2p_n + (p2p_young_boost if is_young else f_p2p_nonyouth) + \
                     (p2p_cent_boost if network_centrality_score > cent_p2p_thresh else f_p2p_lowcent)
        

        # --- 1. IMPROVED CALIBRATION STEP (RISK-WEIGHTED) ---
        # The PROBABILITY of an intense spike is now based on combined risk factors
        Total_Behavioral_Risk = F_burst * F_size * F_inter * F_intra * F_amt * network_centrality_score * channel_hop_score


        
        # --- MODIFIED: Log-Compress the risk to prevent exponential explosion ---

        compressed_total_risk = 1 + np.log1p(Total_Behavioral_Risk)

        
        # --- 2026 OPTIMIZED ACTIVATION LOGIC ---
        
        # 1. Logic Constants (Block 4 Overrides)
        s_log_adj = opt_overrides.get(f"{scenario_type}_risk_log_base_adj", 1.0)

        # 2. Optimized Risk Compression
        # Optuna tunes 's_log_adj' to see how sensitive the spike is to risk
        compressed_total_risk = s_log_adj + np.log1p(Total_Behavioral_Risk)
        



        spike_prob = min(0.8, 0.15 + (compressed_total_risk / 20.0)) # Scaled for compressed value

        s_ceiling = opt_overrides.get(f"{scenario_type}_spike_prob_ceiling", 0.8)
        s_floor   = opt_overrides.get(f"{scenario_type}_spike_prob_floor", 0.15)
        s_divisor = opt_overrides.get(f"{scenario_type}_risk_scaling_divisor", 20.0)

        # 3. Optimized Spike Probability
        # Optuna tunes floor, ceiling, and divisor to find the "Sweet Spot" for Recall
        spike_prob = min(s_ceiling, s_floor + (compressed_total_risk / s_divisor))


        # Higher total risk increases the chance of intense, aggressive behavior
        # Use a scaled probability based on the new combined risk factors

        # --- 2026 HOLISTIC INTENSITY REFACTOR ---
        # Total_Behavioral_Risk incorporates Network, Channel, and Demographic factors
        #spike_prob = min(0.8, 0.15 + (Total_Behavioral_Risk / 100.0))


        if np.random.random() < spike_prob:
            # Use compressed risk for scaling to keep intensity manageable
            # Change the divisor from 10.0 to 20.0 to dampen the intensity
            risk_scaling = 1 + (compressed_total_risk / 20.0)
            
            intensity = np.random.uniform(2.0, 4.0) * risk_scaling
        else:
            intensity = np.random.uniform(0.9, 1.1)
        
        #if np.random.random() < spike_prob:
            # INTEGRATED LOGIC: Aggression scales with the holistic risk profile
            #risk_scaling = 1 + (Total_Behavioral_Risk / 200.0)
            #intensity = np.random.uniform(2.0, 4.0) * risk_scaling
        #else:
            # Baseline protection for 'Normal' profiles
            #intensity = np.random.uniform(0.9, 1.1)


        # --- 2026 CALIBRATED INTENSITY ---
        if np.random.random() < spike_prob:
            # 1. Resolve Magnitude Divisor (Optimizing the 20.0)
            i_baseline = opt_overrides.get(f"{scenario_type}_intensity_baseline", 1.0)
            mag_divisor = opt_overrides.get(f"{scenario_type}_intensity_mag_divisor", 20.0)
            
            risk_scaling = i_baseline + (compressed_total_risk / mag_divisor)
            
            # 2. Resolve High Intensity Range (Optimizing 2.0 and 4.0)
            i_high_min = opt_overrides.get(f"{scenario_type}_intensity_high_min", 2.0)
            i_high_max = opt_overrides.get(f"{scenario_type}_intensity_high_max", 4.0)
            
            intensity = np.random.uniform(i_high_min, i_high_max) * risk_scaling
        else:
            # 3. Resolve Low Intensity Range (Optimizing 0.9 and 1.1)
            i_low_min = opt_overrides.get(f"{scenario_type}_intensity_low_min", 0.9)
            i_low_max = opt_overrides.get(f"{scenario_type}_intensity_low_max", 1.1)
            
            intensity = np.random.uniform(i_low_min, i_low_max)


            
        rows = []
        t_prev = anchor



        # --- MODIFIED: Use Log-compressed factors for loop counts ---
        n_bursts = max(2, int(cfg['n_bursts_base'] * intensity * (1 + np.log1p(F_burst))))
        burst_size = max(5, int(cfg['burst_size_base'] * intensity * (1 + np.log1p(F_size))))


        # --- 2026 OPTIMIZED BURST PHYSICS ---
        nb_min  = opt_overrides.get(f"{scenario_type}_n_bursts_min", 2)
        nb_base = opt_overrides.get(f"{scenario_type}_n_bursts_base_opt", cfg.get('n_bursts_base', 1))
        nb_ladj = opt_overrides.get(f"{scenario_type}_n_bursts_log_adj", 1.0)
        
        bs_min  = opt_overrides.get(f"{scenario_type}_burst_size_min", 5)
        bs_base = opt_overrides.get(f"{scenario_type}_burst_size_base_opt", cfg.get('burst_size_base', 6))
        bs_ladj = opt_overrides.get(f"{scenario_type}_burst_size_log_adj", 1.0)
        
        n_bursts = max(nb_min, int(nb_base * intensity * (nb_ladj + np.log1p(F_burst))))
        burst_size = max(bs_min, int(bs_base * intensity * (bs_ladj + np.log1p(F_size))))
        
                         
        # Determine number of bursts and burst sizes
        #n_bursts = max(2, int(cfg['n_bursts_base'] * intensity * F_burst))
        #burst_size = max(5, int(cfg['burst_size_base'] * intensity * F_size))


        # EMERGENCY BRAKE: Hard cap to ensure 2026 Simulation Stability
        # Lower the Emergency Brake to 500 rows per anchor
        if (n_bursts * burst_size) > 500:
            burst_size = max(5, 500 // n_bursts)
        
        tx_cap  = opt_overrides.get(f"{scenario_type}_total_tx_cap", 500)
        # Cap the simulation to prevent memory/performance explosion
        if (n_bursts * burst_size) > tx_cap:
            burst_size = max(bs_min, int(tx_cap // n_bursts))
        
        
        total_transactions = n_bursts * burst_size

        # --- REFINED LABELING LOGIC (HOLISTIC & BINARY) ---
        if intensity > 1.8 or \
           (total_transactions > 25 and network_centrality_score > 3.0) or \
           (savings_drain_ratio > 0.85):
            current_label = 1 # Suspicious Velocity Spike
        else:
            current_label = 0 # Explicitly set to 0 to prevent label leakage/undefined errors

        # --- 2026 RECONCILED LABELING LOGIC ---
        # Optuna finds the thresholds that best align 'Synthetic Reality' with 'Model Detection'
        l_int_t   = opt_overrides.get(f"{scenario_type}_label_intensity_thresh", 1.8)
        l_tx_t    = opt_overrides.get(f"{scenario_type}_label_tx_count_thresh", 25)
        l_cent_t  = opt_overrides.get(f"{scenario_type}_label_centrality_thresh", 3.0)
        l_drain_t = opt_overrides.get(f"{scenario_type}_label_drain_thresh", 0.85)

        if intensity > l_int_t or \
           (total_transactions > l_tx_t and network_centrality_score > l_cent_t) or \
           (savings_drain_ratio > l_drain_t):
            current_label = 1 
        else:
            current_label = 0
        
        
        # --- EVT RETROFIT: GPD-Driven Extreme Value Sampling ---

        # --- 2026 EVT & MONETARY CALIBRATION (Block 6) ---
        
        # 1. Resolve EVT Parameters from Overrides
        v_thresh = opt_overrides.get(f"{scenario_type}_vs_threshold_pct", 95.0)
        v_conf   = opt_overrides.get(f"{scenario_type}_vs_confidence_level", 0.99)
        v_n      = int(opt_overrides.get(f"{scenario_type}_vs_sample_n", 5))
        
        
        # Now our call inside def generate_events_from_params will run perfectly:
        evt_payload = extreme_value_theory(
            tx_window, 
            scenario_type="velocity_spike",
            #threshold_pct=trial.suggest_float("vs_threshold_pct", 90, 99),
            #confidence_level=trial.suggest_float("vs_confidence_level", 0.99, 0.9999),
            #sample_n=trial.suggest_int("vs_sample_n", 3, 15)
            threshold_pct=v_thresh,
            confidence_level=v_conf,
            sample_n=v_n            
        )
        
        # 1. Access the EVT Params from the pre-computed payload
        spike_data = evt_payload["velocity_spike"]["high_tail_outliers"]
        evt_params = spike_data["params"]

        # BAYESIAN SUGGESTIONS: Calibrated for "Economic Drain" behavior
        # High intensity velocity spikes typically move a high percentage of account value

        # 1. Base Parameter Calculation (Standard Placement)
        # F_amt represents your scaling factors like Income_f * PEP_f
        mu_adj_base = cfg['amt_mu'] * F_amt * (1 + 0.05 * (intensity - 1))

        # 2. Monetary Distribution Physics
        # Tuning the hardcoded 0.05 intensity sensitivity
        
        # --- 2026 OPTIMIZED MONETARY BASES ---
        # 1. Optimize the Config Defaults
        mu_base    = opt_overrides.get(f"{scenario_type}_amt_mu_base_opt", cfg.get('amt_mu', 1100))

        # 2. Optimize the Intensity Physics (Replacing the hardcoded 1 and -1)
        mu_i_intercept = opt_overrides.get(f"{scenario_type}_mu_intensity_intercept", 1.0)
        i_factor = opt_overrides.get(f"{scenario_type}_mu_adj_intensity_factor", 0.05)        
        mu_i_offset    = opt_overrides.get(f"{scenario_type}_mu_intensity_offset", 1.0)
        
        mu_adj_base = mu_base * F_amt * (mu_i_intercept + i_factor * (intensity - mu_i_offset))
        
        
        sigma_base = cfg['amt_sigma']
        
        sigma_base = opt_overrides.get(f"{scenario_type}_amt_sigma_base_opt", cfg.get('amt_sigma', 400))

        
        # 2. Bayesian Strategy for Velocity Spikes
        # We use the same concepts from structuring but apply them to the Spike context
        # mean_shift: pushes the 'burst' amounts higher (e.g., 1.1x to 1.3x)
        # sigma_scale: tightens the burst (e.g., 0.4x to 0.7x) to show mechanical behavior
        tuned_mu = mu_adj_base * trial.suggest_float("vs_spike_mean_shift", 1.05, 1.3)
        tuned_sigma = sigma_base * trial.suggest_float("vs_spike_sigma_contraction", 0.3, 0.8)


        # Tuning the Mean Shift and Sigma Contraction
        m_shift = opt_overrides.get(f"{scenario_type}_vs_spike_mean_shift", 1.1)
        s_cont  = opt_overrides.get(f"{scenario_type}_vs_spike_sigma_contraction", 0.5)

        tuned_mu = mu_adj_base * m_shift
        tuned_sigma = sigma_base * s_cont
        
        

        # 3. Recalculate Bounds for Truncated Normal
        # This ensures the spike stays within the global config limits [amt_min, amt_max]
        lower_bound = cfg['amt_min']
        upper_bound = cfg['amt_max']        

        # 3. Boundary Optimization
        lower_bound = opt_overrides.get(f"{scenario_type}_amt_min_opt", cfg['amt_min'])
        upper_bound = opt_overrides.get(f"{scenario_type}_amt_max_opt", cfg['amt_max'])

        # --- 2026 OPTIMIZED TEMPORAL PHYSICS (Block 7) ---
        b_gap_scale = opt_overrides.get(f"{scenario_type}_burst_gap_scale", 3.0)

        t_gap_scale = opt_overrides.get(f"{scenario_type}_tx_gap_scale", 15.0)
        t_gap_mu    = opt_overrides.get(f"{scenario_type}_tx_gap_log_mu", 0.5)
        t_gap_sigma = opt_overrides.get(f"{scenario_type}_tx_gap_log_sigma", 0.2)
        
        
        # 4. Generate Spiky Transactions
        for _ in range(n_bursts):
            # Time between bursts compressed by F_inter
            #t_burst_start = t_prev + timedelta(hours=float(np.random.exponential(scale=3.0 / F_inter)))
            # Scale the 3-hour gap by BOTH the specific Inter-factor and the overall Intensity
            t_burst_start = t_prev + timedelta(hours=float(np.random.exponential(scale=3.0 / (F_inter * intensity))))


            t_burst_start = t_prev + timedelta(hours=float(np.random.exponential(scale=b_gap_scale / (F_inter * intensity))))

            t_prev_tx = t_burst_start

            # --- NEW CODE: Select channel for this burst based on config diversification prob
            current_channel = np.random.choice(cfg['channel_config']['rails'])


            for _ in range(burst_size):
                # Time between tx within burst compressed by F_intra (True Velocity Spike)
                #delta_min = float(np.random.exponential(scale=15.0 / F_intra) + np.random.lognormal(mean=0.5, sigma=0.2))

                # The 15-minute gap is compressed most aggressively by the Intensity
                delta_min = float(np.random.exponential(scale=15.0 / (F_intra * intensity)) + np.random.lognormal(mean=0.5, sigma=0.2))

                # Time between tx: 15.0m anchor and lognormal params are now optimized
                delta_min = float(np.random.exponential(scale=t_gap_scale / (F_intra * intensity)) + 
                                  np.random.lognormal(mean=t_gap_mu, sigma=t_gap_sigma))                


                
                t_tx = t_prev_tx + timedelta(minutes=delta_min)

                # Truncated Normal for authentic amount clustering
                mu_adj = cfg['amt_mu'] * F_amt * (1 + 0.05 * (intensity - 1))
                sigma = cfg['amt_sigma']
                
                a, b = (cfg['amt_min'] - mu_adj) / sigma, (cfg['amt_max'] - mu_adj) / sigma
                amt = round(truncnorm.rvs(a, b, loc=mu_adj, scale=sigma), 2)


                a, b = (lower_bound - mu_adj_base) / sigma_base, (upper_bound - mu_adj_base) / sigma_base
                amt = round(truncnorm.rvs(a, b, loc=mu_adj_base, scale=sigma_base), 2)


                
                
                # --- 2026 RETROFIT: EVT GATE FOR VELOCITY SPIKES ---

                # --- 2026 RECONCILED EVT LOGIC (Block 7 Overrides) ---
                # Using l_int_t and a new behavioral risk threshold (l_risk_t) from earlier blocks
                l_int_t  = opt_overrides.get(f"{scenario_type}_label_intensity_thresh", 1.8)
                l_risk_t = opt_overrides.get(f"{scenario_type}_compressed_risk_thresh", 5.0)

                
                # TRIGGER: Uses your 'compressed_total_risk' and 'intensity'
                # Velocity EVT triggers when behavioral risk is high and the spike is active
                #if evt_params["sigma_scale"] > 0 and (intensity > 1.8 or compressed_total_risk > 5.0):
                
                if evt_params["sigma_scale"] > 0 and (intensity > l_int_t or compressed_total_risk > l_risk_t):                    
                    
                    # SCALE: Tail spread is widened by Channel Hopping and Network Centrality
                    # This simulates 'complex' laundering where amounts vary wildly across rails
                    dynamic_sigma = evt_params["sigma_scale"] * (1 + (0.05 * channel_hop_score * network_centrality_score))

                    # Optimize the 0.05 hop factor
                    e_hop_f = opt_overrides.get(f"{scenario_type}_evt_sigma_hop_factor", 0.05)
                    dynamic_sigma = evt_params["sigma_scale"] * (1 + (e_hop_f * channel_hop_score * network_centrality_score)

                    
                    # DEPTH: Deeper tails for higher Savings Drain (moving larger chunks of wealth)
                    tail_depth = min(0.995, 0.2 + (savings_drain_ratio / 2.0))

                    # Optimize the tail depth physics (0.995, 0.2, 2.0)

                    e_tail_ceil  = opt_overrides.get(f"{scenario_type}_evt_tail_ceiling", 0.995)                               
                    e_tail_floor = opt_overrides.get(f"{scenario_type}_evt_tail_floor", 0.2)
                    e_tail_drain = opt_overrides.get(f"{scenario_type}_evt_tail_drain_factor", 2.0)

                    tail_depth = min(e_tail_ceil, e_tail_floor + (savings_drain_ratio / e_tail_drain))                                 
                               
                    u_sample = np.random.uniform(0.5, tail_depth)
                    
                    # EVT Exceedance
                    exceedance = genpareto.ppf(u_sample, evt_params["xi_shape"], loc=0, scale=dynamic_sigma)
                    #amt = round(min(upper_bound, evt_params["threshold_u"] + exceedance), 2)
                    amt = round(evt_params["threshold_u"] + exceedance, 2)

                
                else:
                    # FALLBACK: Bayesian Baseline
                    # Higher 'mean_translation' pushes the burst volume up, 
                    # while 'sigma_contraction' makes it look like repetitive automated spikes

                    # 1. Resolve Safety Floor (Optimizing the 0.01)
                    s_floor = opt_overrides.get(f"{scenario_type}_sigma_min_floor", 0.01)
                        
                    a_spike = (lower_bound - tuned_mu) / max(0.01, tuned_sigma)
                    b_spike = (upper_bound - tuned_mu) / max(0.01, tuned_sigma)

                    a_spike = (lower_bound - tuned_mu) / max(s_floor, tuned_sigma)
                    b_spike = (upper_bound - tuned_mu) / max(s_floor, tuned_sigma)
                            
                    # 4. Generate "Professionalized" Spike Amount
                    # This avoids the "clipping" artifacts and creates a realistic 'tight' cluster of high-value trades
                    amt = round(truncnorm.rvs(a_spike, b_spike, loc=tuned_mu, scale=tuned_sigma), 2)                    
                
                ##tx_type = random.choices(
                    ##["DEBIT", "CREDIT", "P2P"],
                    ##weights=[0.7 * F_type_debit, 0.2 * F_type_credit, 0.1 * F_type_p2p]
                ##)[0]


                #tx_type = random.choices(
                    #["DEBIT", "CREDIT"],
                    #weights=[0.7 * F_type_debit, 0.3 * F_type_credit]
                #)[0]


                # ---------------------------------------------------------
                # SUPERSEDING RULE: Rail-Hopping logic for 2026 AML 
                # ---------------------------------------------------------
                chan = cfg['channel_config']

                # CHANGE THIS LINE:
                # is_rail_hop = np.random.random() < chan['hop_diversification_prob']
                
                # TO THIS (Matches your JSON key 'diversification_prob'):
                is_rail_hop = np.random.random() < chan['diversification_prob']
                            
                d_prob = opt_overrides.get(f"{scenario_type}_diversification_prob_opt", chan.get('diversification_prob', 0.65))
                is_rail_hop = np.random.random() < d_prob

                
                if is_rail_hop:
                    # High-intensity signal: Pick a high-risk rail
                    tx_type = np.random.choice(chan['rails'])
                else:
                    # Status quo: standard fiat directionality
                    tx_type = np.random.choice(["CREDIT", "DEBIT"], p=[0.5, 0.5])

                    # Optimized Baseline Weights (Block 8)
                    p_credit = opt_overrides.get(f"{scenario_type}_tx_type_credit_p", 0.5)
                    p_debit  = opt_overrides.get(f"{scenario_type}_tx_type_debit_p", 0.5)
                    p_sum = p_credit + p_debit
                    tx_type = np.random.choice(["CREDIT", "DEBIT"], p=[p_credit/p_sum, p_debit/p_sum])
                
                #rows.append(_assemble_tx(acct, party_key, t_tx, amt, tx_type, cfg))
                #rows.append(_assemble_tx(acct, party_key, ref_date, t_tx, amt, tx_type, cfg, label=current_label))

                # We need to pass the channel information into the final transaction object
                # Assuming _assemble_tx can handle an extra 'channel' parameter
                rows.append(_assemble_tx(acct, party_key, ref_date, t_tx, amt, tx_type, cfg, label=current_label))

                t_prev_tx = t_tx
    
            # Micro-transactions after burst
            #rows.extend(_generate_micro_tx(t_prev_tx, amt, cfg))
            
            # Micro-transactions now scale with F_micro from the new JSON location
            #F_micro_val = (dw["F_micro"]["PEP"] if is_pep else 1.0) * (1 + risk_score * dw["F_micro"]["risk_score"])


            # --- 2026 CORRECTION: PROTECTED MICRO-TX SCALING ---
            # We scale the COUNT of micro-tx with risk, but CAP the amount 
            # to ensure they remain 'micro' (under $150) and don't distort precision.
            F_micro_val = (dw["F_micro"]["PEP"] if is_pep else 1.0) * (1 + risk_score * dw["F_micro"]["risk_score"])

            # --- 2026 OPTIMIZED MICRO-TRANSACTION LOGIC (Block 9) ---
            
            # 1. Resolve Multipliers and Neutrals
            #f_mic_pep   = opt_overrides.get(f"{scenario_type}_F_micro_PEP", dw["F_micro"].get("PEP", 0.5))
            #f_mic_pep_n = opt_overrides.get(f"{scenario_type}_F_micro_PEP_neutral", 1.0)
            #f_mic_rss   = opt_overrides.get(f"{scenario_type}_F_micro_risk_scale", dw["F_micro"].get("risk_score", 0.3))
    
            #F_micro_val = (f_mic_pep if is_pep else f_mic_pep_n) * (1 + risk_score * f_mic_rss)

                        
            # Use a conservative fraction (e.g. 5%) of the main amt, but never exceeding a realistic noise ceiling
            micro_amt_base = min(amt * 0.05 * F_micro_val, 150.0)

            # 2. Amount and Noise Logic (Replacing 0.05 and 150.0)
            #m_fraction = opt_overrides.get(f"{scenario_type}_micro_amt_fraction", 0.05)
            #m_ceiling  = opt_overrides.get(f"{scenario_type}_micro_noise_ceiling", 150.0)
            #micro_amt_base = min(amt * m_fraction * F_micro_val, m_ceiling)
                        
            # Ensure n_micro also reflects the burst intensity
            n_micro_burst = max(1, int(2 * intensity * F_micro_val))

            # 3. Frequency and Jitter (Replacing 2, 0.7, and 1.2)
            #m_n_base = opt_overrides.get(f"{scenario_type}_micro_n_base", 2.0)
            #n_micro_burst = max(1, int(m_n_base * intensity * F_micro_val))

            #m_jit_min = opt_overrides.get(f"{scenario_type}_micro_jitter_min", 0.7)
            #m_jit_max = opt_overrides.get(f"{scenario_type}_micro_jitter_max", 1.2)

                             
            for _ in range(n_micro_burst):
                # Jitter the micro-amount so they aren't all identical
                micro_amt = round(micro_amt_base * np.random.uniform(0.7, 1.2), 2)

                #micro_amt = round(micro_amt_base * np.random.uniform(m_jit_min, m_jit_max), 2)
                #rows.extend(_generate_micro_tx(t_prev_tx, micro_amt, config=cfg, ref_date=ref_date, label=current_label, channel=current_channel))

            
            ###rows.extend(_generate_micro_tx(t_prev_tx, amt * F_micro_val, config, ref_date=ref_date, label=current_label))
            ###rows.extend(_generate_micro_tx(t_prev_tx, amt * F_micro_val, config=cfg, ref_date=ref_date, label=current_label))

            ###rows.extend(_generate_micro_tx(t_prev_tx, amt * F_micro_val, config=cfg, ref_date=ref_date, label=current_label, channel=current_channel))
            
            t_prev = t_prev_tx

    
    
    elif scenario_type == "layering":
        
        cfg = scenario_config['layering']
        dw = cfg['demographics']

        # 1. Sample demographics (The Critical Handshake)
        entity_type = random.choices(list(dw['entity_type_weights'].keys()), weights=list(dw['entity_type_weights'].values()))[0] ## Here is the missing line: Required for lookup
        risk_score = random.choices(list(dw['risk_score_weights'].keys()), weights=list(dw['risk_score_weights'].values()))[0] ## Here is the missing line: Required for risk logic
        nationality = random.choices(list(dw['nationality_weights'].keys()), weights=list(dw['nationality_weights'].values()))[0] ## Here is the missing line: Required for offshore gap logic
        pep_status = random.random() < dw['pep_status_prob']
        industry = random.choice(dw.get('industry_options', ['other'])) if entity_type == "corporate" else None
        
        # --- 0. EXTRACT 3D PARAMETERS ---
        topo, chan, econ = cfg['network_topology'], cfg['channel_config'], cfg['economic_sensibility']

        # --- 2. STREAMLINED 3D RISK MODIFIER ---
        # Consolidates Topology (Centrality), Diversity (Switching), and Economic (Escalation)
        total_risk_mod = (topo['centrality_boost'] if risk_score == "high" else 1.0) * \
                         (chan['rail_switch_intensity'] / 2.0) * \
                         (econ['staged_escalation_factor'] if risk_score != "low" else 1.0)


        # Helper function to handle flattened dictionary from Bayesian Optimization
        def get_factor(source, actor_val):
            # If it's the expected dict, do a lookup
            if isinstance(source, dict):
                return source.get(str(actor_val), 1.0)
            # If it's a flattened number (like your debug showed), use it directly
            if isinstance(source, (int, float)):
                return float(source)
            return 1.0
            
        
        # --- THE 2026 LAYERING BRIDGE ---
        # 1. Align Actor with the "Structuring" Style (0, 1, 2 indices)
        actor = {
            "EntityType":  0 if entity_type == "individual" else (1 if entity_type == "sme" else 2),
            "RiskScore":   0 if risk_score == "low" else (1 if risk_score == "medium" else 2),
            "Nationality": 0 if nationality == "domestic" else (1 if nationality == "mid-risk" else 2),
            "PEP":         1 if pep_status else 0
        }




        def dynamic_sample(field_name, sub_path='demographics'):
            """
            Categorical sampler that handles weight dictionaries and Optuna overrides.
            """
            prob_key = f"{scenario_type}_{field_name}_probs"
            
            # Layering uses cfg['demographics'], aliased as 'dw'
            # Use clean key (e.g., 'industry_weights') for the static config lookup
            clean_key = f"{field_name}_weights"
            
            # 1. Get baseline weights from demographics (dw)
            # Extract .values() if it's a dict (individual weights are often named)
            baseline_dict = dw.get(clean_key, {})
            default_probs = list(baseline_dict.values()) if isinstance(baseline_dict, dict) else baseline_dict
        
            # 2. Prioritize Bayesian shifts from Optuna
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            return np.random.choice(len(p_vector), p=p_vector)


        # Execution in generate_events_from_params for Layering
        actor = {
            "EntityType":  dynamic_sample("entity_type"),
            "RiskScore":   dynamic_sample("risk_score"),
            "Nationality": dynamic_sample("nationality"),
            "PEP":         1 if np.random.random() < opt_overrides.get(
                               f"{scenario_type}_pep_status_prob", 
                               dw.get('pep_status_prob', 0.05)
                           ) else 0,
                           
            # --- NEW: INDUSTRY VARIATION ---
            # We sample industry index (0, 1, 2...) from optimized weights
            "Industry":    dynamic_sample("industry")
        }



        # 2. Compound Multipliers (Now BO can tune these via trial.suggest_float)
        Entity_f = get_factor(df_cfg.get('EntityType'), actor['EntityType'])
        Risk_f   = get_factor(df_cfg.get('RiskScore'), actor['RiskScore'])
        Juris_f  = get_factor(df_cfg.get('Nationality'), actor['Nationality'])
        PEP_f    = get_factor(df_cfg.get('PEP'), actor['PEP'])

        # Add this line before Static_Risk calculation
        # Use the same 'industry' variable you sampled in the Handshake
        industry_idx = dw.get('industry_options', []).index(industry) if industry in dw.get('industry_options', []) else 3        Industry_f = get_factor(df_cfg.get('Industry'), industry_idx)


        Static_Risk = Entity_f * Juris_f * PEP_f * Industry_f




        # --- RECONCILED LOOKUPS (LAYERING) ---
        # 1. Check opt_overrides for a Bayesian suggestion (e.g., 'layering_Entity_f')
        # 2. Fall back to get_factor(df_cfg...) if no suggestion exists
        
        Entity_f   = opt_overrides.get(f"{scenario_type}_Entity_f", 
                                       get_factor(df_cfg.get('EntityType'), actor['EntityType']))
        
        Risk_f     = opt_overrides.get(f"{scenario_type}_Risk_f", 
                                       get_factor(df_cfg.get('RiskScore'), actor['RiskScore']))
        
        Juris_f    = opt_overrides.get(f"{scenario_type}_Juris_f", 
                                       get_factor(df_cfg.get('Nationality'), actor['Nationality']))
        
        PEP_f      = opt_overrides.get(f"{scenario_type}_PEP_f", 
                                       get_factor(df_cfg.get('PEP'), actor['PEP']))
        
        # Industry Lookup: Uses the index already stored in actor['Industry'] 
        # Placeholder -1 is handled by get_factor returning 1.0
        Industry_f = opt_overrides.get(f"{scenario_type}_Industry_f", 
                                       get_factor(df_cfg.get('Industry'), actor['Industry']))
        
        # Final Product
        Static_Risk = Entity_f * Risk_f * Juris_f * PEP_f * Industry_f



        # 3. The 2026 Layering "Total_Behavioral_Risk" Bridge
        # This now matches the complexity of your Structuring scenario
        # Static/Demographic Foundation
        
        # Behavioral/Economic Integration
        # rail_switch_intensity and staged_escalation_factor act as 'Complexity Multipliers'
        Total_Behavioral_Risk = (Static_Risk) * \
                                (topo['centrality_boost'] if risk_score == "high" else 1.0) * \
                                (chan['rail_switch_intensity'] / 2.0) * \
                                (econ['staged_escalation_factor'])


        # --- 1. TOTAL BEHAVIORAL RISK (Reconciled) ---
        # Sourcing hardcoded: topo centrality (1.0 fallback), rail divisor (2.0), escalation
        # Now optimizing the rail_switch_intensity (Prior: chan['rail_switch_intensity'])
        rail_switch_intensity = opt_overrides.get(f"{scenario_type}_rail_switch_int", chan.get('rail_switch_intensity', 1.0))
        rail_div        = opt_overrides.get(f"{scenario_type}_rail_div_factor", 2.0)
        centrality_boost         = opt_overrides.get(f"{scenario_type}_centrality_boost", 1.0 if risk_score == "high" else 1.0)
        staged_escalation_factor      = opt_overrides.get(f"{scenario_type}_staged_escalation_f", econ.get('staged_escalation_factor', 1.0))
        
        Total_Behavioral_Risk = (Static_Risk) * centrality_boost * (rail_switch_intensity / rail_div) * staged_escalation_factor


        # --- STANDALONE INTENSITY CALCULATION ---
        # This ensures structural risk (3D) drives the behavioral signal
        #if np.random.random() < 0.30:
            #intensity = np.random.uniform(2.0, 4.0) * total_risk_mod
        #else:
            # Baseline intensity still reflects 50% of the structural risk
            #intensity = 1.0 * (total_risk_mod * 0.5)
        
        # 2026 Safeguard: Cap intensity to prevent mathematical overflow in amt logic
        #intensity = max(0.5, min(8.0, round(intensity, 2)))


        # --- 2026 ENHANCED INTENSITY CALCULATION ---
        # Total_Behavioral_Risk now acts as the 'Primary Engine' for all behavioral signals
        if np.random.random() < 0.30:
            # 30% chance of a high-intensity 'Criminal Burst' scaled by structural risk
            intensity = np.random.uniform(2.0, 4.0) * Total_Behavioral_Risk
        else:
            # Baseline intensity reflects 50% of the risk, ensuring 'High-Risk Normal' behavior
            intensity = 1.0 * (Total_Behavioral_Risk * 0.5)

        # 2026 Safeguard: Cap intensity to prevent mathematical overflow in amt logic
        intensity = max(0.5, min(8.0, round(intensity, 2)))
        


        # --- 2026 ENHANCED INTENSITY CALCULATION (Reconciled) ---
        # Sourcing hardcoded: 0.30 chance, 2.0-4.0 burst, 0.5 norm scale
        burst_p    = opt_overrides.get(f"{scenario_type}_burst_p", 0.30)
        burst_low  = opt_overrides.get(f"{scenario_type}_burst_low", 2.0)
        burst_high = opt_overrides.get(f"{scenario_type}_burst_high", 4.0)
        norm_scale = opt_overrides.get(f"{scenario_type}_norm_scale", 0.5)

        if np.random.random() < burst_p:
            # 30% chance (tunable) of a high-intensity 'Criminal Burst'
            intensity = np.random.uniform(burst_low, burst_high) * Total_Behavioral_Risk
        else:
            # Baseline intensity reflects scaled risk
            intensity = 1.0 * (Total_Behavioral_Risk * norm_scale)

        # 2026 Safeguards: Sourcing 0.5, 8.0, and round to 2
        int_floor   = opt_overrides.get(f"{scenario_type}_int_floor", 0.5)
        int_ceiling = opt_overrides.get(f"{scenario_type}_int_ceiling", 8.0)
        
        # Capping intensity to prevent mathematical overflow
        intensity = max(int_floor, min(int_ceiling, round(intensity, 2)))



        # 2. COMPOUND HOPS (The Path Signal)
        # We multiply factors to reach "huge" path lengths
        hop_mult = (dw['n_hops_multiplier'].get(entity_type, 1.0) * 
                    (dw['n_hops_multiplier']['high_risk'] if risk_score == "high" else 1.0))
        
        n_hops = max(3, int(cfg['n_hops_base'] * intensity * hop_mult))

        # --- 3. COMPOUND HOPS (The Path Signal) ---
        # Sourcing hardcoded: 3 (min hops), 1.0 (baseline multiplier)
        hop_min       = int(opt_overrides.get(f"{scenario_type}_hop_min", 3))
        hop_base_mult = opt_overrides.get(f"{scenario_type}_hop_base_mult", 1.0)

        hop_mult = (dw['n_hops_multiplier'].get(entity_type, hop_base_mult) * 
                    (dw['n_hops_multiplier']['high_risk'] if risk_score == "high" else hop_base_mult))
        
        n_hops = max(hop_min, int(cfg['n_hops_base'] * intensity * hop_mult))


        # Intensity is now a product of baseline randomness and the 3D risk modifier
        if (np.random.random() > (1.8 / total_risk_mod)) or (n_hops > 6) or (risk_score == "high" and intensity > 1.4):
        
            current_label = 1 
        else:
            current_label = 0


        # --- 4. LABEL ASSIGNMENT (Reconciled) ---
        # 1. Sourcing the risk divisor (Original: 1.8)
        label_risk_div    = opt_overrides.get(f"{scenario_type}_label_risk_div", 1.8)
        
        # 2. Sourcing the hop count limit (Original: 6)
        hop_threshold     = opt_overrides.get(f"{scenario_type}_hop_threshold", 6)
        
        # 3. Sourcing the high-risk intensity trigger (Original: 1.4)
        label_trigger_int = opt_overrides.get(f"{scenario_type}_label_trigger_int", 1.4)
        
        # Execution of the multi-trigger labeling logic
        if (np.random.random() > (label_risk_div / Total_Behavioral_Risk)) or \
           (n_hops > hop_threshold) or \
           (risk_score == "high" and intensity > label_trigger_int):
        
            current_label = 1 
        else:
            current_label = 0



        # 3. COMPOUND AMOUNTS (The Volume Signal)
        amt_mult = dw['amt_multiplier'].get("default", 1.0)
        if entity_type == "corporate" and industry == "finance":
            amt_mult *= dw['amt_multiplier']['corporate_finance']
        elif entity_type == "SME":
            amt_mult *= dw['amt_multiplier']['SME']
        if pep_status:
            amt_mult *= dw['amt_multiplier']['PEP']

        # Use truncnorm for authentic initial amount
        mu_amt = cfg['amt_min'] * amt_mult * intensity
        sigma_amt = (cfg['amt_max'] - cfg['amt_min']) / 4
        a, b = (cfg['amt_min'] - mu_amt) / sigma_amt, (cfg['amt_max']*amt_mult*2 - mu_amt) / sigma_amt
        base_amount = truncnorm.rvs(a, b, loc=mu_amt, scale=sigma_amt)


        # --- 3. COMPOUND AMOUNTS (The Volume Signal) ---
        # Sourcing hardcoded: 1.0 (default mult), 4.0 (sigma divisor), 2.0 (upper bound scale)
        amt_mult_base = opt_overrides.get(f"{scenario_type}_amt_mult_base", 1.0)
        amt_sigma_div = opt_overrides.get(f"{scenario_type}_amt_sigma_div", 4.0)
        amt_max_scale = opt_overrides.get(f"{scenario_type}_amt_max_scale", 2.0)

        amt_mult = dw['amt_multiplier'].get("default", amt_mult_base)
        if entity_type == "corporate" and industry == "finance":
            amt_mult *= dw['amt_multiplier']['corporate_finance']
        elif entity_type == "SME":
            amt_mult *= dw['amt_multiplier']['SME']
        if pep_status:
            amt_mult *= dw['amt_multiplier']['PEP']

        # Use truncnorm for authentic initial amount
        mu_amt = cfg['amt_min'] * amt_mult * intensity
        sigma_amt = (cfg['amt_max'] - cfg['amt_min']) / amt_sigma_div

        # Boundary calculation for truncnorm
        a = (cfg['amt_min'] - mu_amt) / sigma_amt
        b = (cfg['amt_max'] * amt_mult * amt_max_scale - mu_amt) / sigma_amt
        base_amount = truncnorm.rvs(a, b, loc=mu_amt, scale=sigma_amt)


        # 4. GENERATE INTERMEDIARY LAYERS
        # --- 3. GENERATE LAYERS (Topology + Diversity Implementation) ---
        #t_prev = anchor
        # active_funds list implements NETWORK TOPOLOGY (Branching)
        #active_funds = [{"amt": base_amount, "time": t_prev}]

        # --- 2026 CONSTRUCTIVE FIX: MACRO-ANCHOR OFFSET ---
        # Instead of starting exactly at 'anchor', we add a small initial jitter
        # that is SHORTER for high-intensity/high-risk actors (Urgency)
        start_jitter = np.random.uniform(0, 12.0) / (intensity * max(0.5, total_risk_mod))
        t_start = anchor + timedelta(hours=start_jitter)

        # --- 4. MACRO-ANCHOR OFFSET (The Urgency Signal) ---
        # Sourcing hardcoded: 12.0 hours max jitter, 0.5 urgency floor
        jitter_base  = opt_overrides.get(f"{scenario_type}_jitter_base", 12.0)
        urgency_floor = opt_overrides.get(f"{scenario_type}_urgency_floor", 0.5)

        # High-intensity/High-risk actors move faster (shorter jitter)
        start_jitter = np.random.uniform(0, jitter_base) / (intensity * max(urgency_floor, Total_Behavioral_Risk))
        t_start = anchor + timedelta(hours=start_jitter)
                                     
        # active_funds now starts with the jittered time
        active_funds = [{"amt": base_amount, "time": t_start}]

        # Now our call inside def generate_events_from_params will run perfectly:
        evt_payload = extreme_value_theory(
            tx_window, 
            scenario_type="layering",
            threshold_pct=trial.suggest_float("layer_threshold_pct", 90, 99),
            confidence_level=trial.suggest_float("layer_confidence_level", 0.99, 0.9999),
            sample_n=trial.suggest_int("layer_sample_n", 3, 15)
        )


        # --- 1. EVT RETROFIT: GPD-Driven Extreme Value Sampling ---
        # Sourcing from opt_overrides (Suggested Upstream in Objective)
        # No Enqueue used for 'layer_' parameters to allow pure discovery
        l_threshold = opt_overrides.get(f"{scenario_type}_layer_threshold_pct", 95.0)
        l_conf      = opt_overrides.get(f"{scenario_type}_layer_confidence_level", 0.999)
        l_sample    = int(opt_overrides.get(f"{scenario_type}_layer_sample_n", 5))

        evt_payload = extreme_value_theory(
            tx_window, 
            scenario_type="layering",
            threshold_pct=l_threshold,
            confidence_level=l_conf,
            sample_n=l_sample
        )

        # 1. Access the EVT Params from the pre-computed payload
        layering_data = evt_payload["layering"]["high_tail_outliers"]
        evt_params = layering_data["params"]

        # suggestion: define upper_bound globally from config to prevent NameError
        lower_bound = cfg['amt_min']
        upper_bound = cfg['amt_max'] 

        # SUGGESTED BAYESIAN STRATEGY: Define once to ensure entity consistency
        lay_m_shift = trial.suggest_float("layer_mean_retention", 0.98, 1.0)
        lay_s_scale = trial.suggest_float("layer_sigma_precision", 0.2, 0.5)

        # --- 3. BAYESIAN STRATEGY: Fund Retention & Precision ---
        # These determine how much money is "kept" vs "passed on" in the layers
        lay_m_shift = opt_overrides.get(f"{scenario_type}_layer_mean_retention", 0.99)
        lay_s_scale = opt_overrides.get(f"{scenario_type}_layer_sigma_precision", 0.3)

        # 4. GENERATE INTERMEDIARY LAYERS
        rows = []
        for d in range(n_hops):
            next_layer_funds = []
            for fund in active_funds:

                # --- 1. VELOCITY & GAP COMPRESSION ---
                # Sourcing hardcoded: diversification prob, bypass mult, decay mult
                hop_div_p      = opt_overrides.get(f"{scenario_type}_hop_div_p", chan.get('hop_diversification_prob', 0.2))
                lat_bypass_f   = opt_overrides.get(f"{scenario_type}_lat_bypass_f", chan.get('latency_bypass_multiplier', 1.0))
                vel_decay_indiv = opt_overrides.get(f"{scenario_type}_vel_decay_indiv", econ.get('velocity_decay_on_low_wealth', 1.0))
                
                # CHANNEL DIVERSITY: Rail hopping vs legacy
                is_rail_hop = random.random() < chan['hop_diversification_prob']
                v_bypass = chan['latency_bypass_multiplier'] if is_rail_hop else 1.0
                
                # ECONOMIC SENSIBILITY: Velocity decay based on actor wealth/type
                w_decay = econ['velocity_decay_on_low_wealth'] if entity_type == "individual" else 1.0

                is_rail_hop = np.random.random() < hop_div_p
                v_bypass = lat_bypass_f if is_rail_hop else 1.0
                w_decay = vel_decay_indiv if entity_type == "individual" else 1.0


                gap_adj = dw['gap_compression'].get(entity_type, 1.0) * v_bypass * w_decay
                
                gap_comp_base = opt_overrides.get(f"{scenario_type}_gap_comp_base", dw['gap_compression'].get(entity_type, 1.0))
                gap_adj = gap_comp_base * v_bypass * w_decay                
                
                # GAP COMPRESSION: High-risk layering happens FAST (Velocity Gain)
                
                if nationality == "high_risk_offshore":
                    gap_adj *= dw['gap_compression']['high_risk_offshore']


                # --- 2026 REFINEMENT: HOLISTIC COMPRESSION ---
                # We multiply the component 'gap_adj' by 'intensity' and 'Total_Behavioral_Risk'   #'total_risk_mod'
                # This ensures Cat 3 (Suspicious) is significantly faster than Cat 2 (Normal)
                #holistic_compression = gap_adj * intensity * max(0.5, total_risk_mod)
                holistic_compression = gap_adj * intensity * max(0.5, Total_Behavioral_Risk)

                # --- HOLISTIC COMPRESSION ---
                # Sourcing hardcoded floor: 0.5
                h_comp_floor = opt_overrides.get(f"{scenario_type}_holistic_comp_floor", 0.5)
                holistic_compression = gap_adj * intensity * max(h_comp_floor, Total_Behavioral_Risk)

                
                #t_new = fund['time'] + timedelta(hours=float(np.random.exponential(scale=cfg['hop_gap_hours_scale'] / max(1e-6, gap_adj)   )))

                # t_new now uses the full integrated risk profile
                t_new = fund['time'] + timedelta(hours=float(
                    np.random.exponential(scale=cfg['hop_gap_hours_scale'] / max(1e-6, holistic_compression))
                ))

                
                # ECONOMIC SENSIBILITY: Balance retention (Mule Fee)
                # --- CORRECTED FLOW CALCULATION ---
                # 1. Base flow after balance retention (The "Mule Tax")
                flow_amt = fund['amt'] * (1 - econ['balance_retention_ratio'])

                # --- 2. ECONOMIC FRICTION (Flow Calculation) ---
                # Sourcing hardcoded retention ratio
                retention_ratio = opt_overrides.get(f"{scenario_type}_bal_retention_ratio", econ.get('balance_retention_ratio', 0.05))
                flow_amt = fund['amt'] * (1 - retention_ratio)
                
                
                # 2. Apply DECAY: High-risk paths lose more value (fees/layering costs)
                decay_range = dw['forward_amt_decay']['high_risk'] if risk_score == "high" else dw['forward_amt_decay']['default']
                #flow_amt *= np.random.uniform(*decay_range)

                # 3. Final jitter for realistic transaction fragmentation
                #flow_amt *= np.random.uniform(0.95, 0.99)

                # Sourcing hardcoded decay ranges
                d_low  = opt_overrides.get(f"{scenario_type}_decay_low", decay_range[0])
                d_high = opt_overrides.get(f"{scenario_type}_decay_high", decay_range[1])
                
                # One single, clean uniform sample for realistic fragmentation
                flow_amt *= np.random.uniform(d_low, d_high)


                # --- 1. EVT SAMPLING (The Outlier Logic) ---
                # Sourcing hardcoded: 2.5 (cutoff), 0.1 (sigma boost), 0.3 (tail floor)
                evt_cutoff   = opt_overrides.get(f"{scenario_type}_evt_int_cutoff", 2.5)
                evt_s_boost  = opt_overrides.get(f"{scenario_type}_evt_sigma_boost", 0.1)
                td_floor     = opt_overrides.get(f"{scenario_type}_tail_depth_floor", 0.3)
                ts_min       = opt_overrides.get(f"{scenario_type}_tail_depth_min", 0.5)                
                
                # --- 2026 HYBRID AMOUNT LOGIC ---
                # Trigger EVT for high intensity or high-risk offshore diversions
                #if evt_params["sigma_scale"] > 0 and (intensity > 2.5 or nationality == "high_risk_offshore"):

                if evt_params["sigma_scale"] > 0 and (intensity > evt_cutoff or nationality == "high_risk_offshore"):
                    # EVT Logic: Extreme 'Peeling' or Diversion Outliers
                    
                    #dynamic_sigma = evt_params["sigma_scale"] * (1 + (0.1 * total_risk_mod))   
                    dynamic_sigma = evt_params["sigma_scale"] * (1 + (0.1 * Total_Behavioral_Risk))
                    
                    dynamic_sigma = evt_params["sigma_scale"] * (1 + (evt_s_boost * Total_Behavioral_Risk))
                    
                    # Tail depth increases with the number of hops (staged escalation)
                    tail_depth = min(0.998, 0.3 + (d / float(n_hops)))

                    tail_depth = min(0.998, td_floor + (d / float(n_hops)))
                    
                    u_sample = np.random.uniform(0.5, tail_depth)

                    u_sample = np.random.uniform(ts_min, tail_depth)
                    
                    exceedance = genpareto.ppf(u_sample, evt_params["xi_shape"], loc=0, scale=dynamic_sigma)
                    # For layering, outliers represent 'leaked' funds or high-value diversions
                    #amt = round(min(upper_bound, flow_amt + exceedance), 2)
                    amt = round(flow_amt + exceedance, 2)
                    
                else:
                    # Bayesian Baseline: Mechanical Intermediary Transfer
                    # We use the pre-calculated strategy to keep amounts consistent across hops

                    sigma_p   = opt_overrides.get(f"{scenario_type}_amt_tuned_sigma_p", 0.05)
                    sigma_min = opt_overrides.get(f"{scenario_type}_amt_sigma_min", 0.01)                    
                    
                    tuned_mu = flow_amt * lay_m_shift
                    
                    tuned_sigma = (flow_amt * 0.05) * lay_s_scale # 5% baseline sigma contracted by strategy
                    
                    tuned_sigma = (flow_amt * sigma_p) * lay_s_scale 
                    
                    a_lay = (cfg['amt_min'] - tuned_mu) / max(0.01, tuned_sigma)                    
                    b_lay = (upper_bound - tuned_mu) / max(0.01, tuned_sigma)

                    a_lay = (cfg['amt_min'] - tuned_mu) / max(sigma_min, tuned_sigma)
                    b_lay = (upper_bound - tuned_mu) / max(sigma_min, tuned_sigma)
                    
                    amt = round(truncnorm.rvs(a_lay, b_lay, loc=tuned_mu, scale=tuned_sigma), 2)
                
                
                # 4. Update the tracker for the next hop (Critical for propagation)
                #next_layer_funds.append({"amt": flow_amt, "time": t_new})

    
                # TX TYPE: Replaces linear weight logic with optimized weights
                #tx_weights = dw['tx_type_weights']['high_risk_offshore'] if nationality == "high_risk_offshore" else dw['tx_type_weights']['default']
                ###tx_type = random.choices(["WIRE", "TRANSFER", "P2P"], weights=tx_weights)[0]
                #normalized_weights = [w / sum(tx_weights) for w in tx_weights]
                #tx_type = random.choices(["WIRE", "TRANSFER", "P2P"], weights=normalized_weights)[0]


                # ---------------------------------------------------------
                # SUPERSEDING RULE: Rail-Hopping logic for 2026 AML 
                # ---------------------------------------------------------
                chan = cfg['channel_config']

                
                is_rail_hop = np.random.random() < chan['hop_diversification_prob']

                # Using override for rail hop probability (matches chan['hop_diversification_prob'])
                rail_p = opt_overrides.get(f"{scenario_type}_rail_p_override", chan_cfg.get('hop_diversification_prob', 0.2))
                
                #if is_rail_hop:
                if np.random.random() < rail_p:
                    # High-intensity signal: Pick a high-risk rail
                    tx_type = np.random.choice(chan['rails'])
                else:
                
                    # --- DEFENSIVE DIRECTION LOOKUP (2026 BO-SAFE) ---
                    # 1. Determine the key based on actor demographics
                    prob_key = "high_risk_offshore" if nationality == "high_risk_offshore" else ("corporate" if entity_type == "corporate" else "default")
                    
                    # 2. Fetch raw weights (which might be raw BO outputs)

                    # Change this:
                    #raw_weights = dir_map[prob_key]
                    
                    # To this:
                    raw_weights = dw['dir_map'][prob_key]

                    # Sourcing weights: Prioritize Optuna shifts (e.g., 'layering_dist_dir_corporate')
                    # Fallback to dw['dir_map'] from the static config
                    raw_weights = opt_overrides.get(f"{scenario_type}_dist_dir_{prob_key}", dw['dir_map'][prob_key])

                    
                    # 3. NORMALIZE: Ensures probabilities sum to 1.0 even under Bayesian drift
                    # This prevents np.random.choice from crashing if the sum is 0.999 or 1.001
                    normalized_probs = [w / sum(raw_weights) for w in raw_weights]
                    
                    # 4. SELECT: CREDIT (Index 0), DEBIT (Index 1)
                    tx_type = np.random.choice(['CREDIT', 'DEBIT'], p=normalized_probs)

                
                # Record transaction safely
                offshore_pool = _HIGH_RISK + _ASIA + _MAJOR + ["LOCAL"]  # always non-empty
                domestic_pool = _ASIA + _MAJOR
                cp_pool = offshore_pool if random.random() < 0.5 else domestic_pool

                # --- 4. COUNTERPARTY POOLING ---
                cp_offshore_p = opt_overrides.get(f"{scenario_type}_cp_offshore_p", 0.5)
                cp_pool = (_HIGH_RISK + _ASIA + _MAJOR + ["LOCAL"]) if np.random.random() < cp_offshore_p else (_ASIA + _MAJOR)
                
                # --- 4. NETWORK TOPOLOGY (Integrated Fan-Out / Fan-In) ---
                
                # --- 4. NETWORK TOPOLOGY (Reconciled) ---
                mule_density = opt_overrides.get(f"{scenario_type}_mule_density", topo.get('mule_cluster_density', 0.4))
                f_out_ratio  = int(opt_overrides.get(f"{scenario_type}_fan_out_ratio", topo.get('fan_out_ratio', 2)))
                f_in_ratio   = opt_overrides.get(f"{scenario_type}_fan_in_ratio", topo.get('fan_in_ratio', 0.3))

                
                # A. FAN-OUT: Split one fund into multiple transactions at the start
                #if d == 0 and random.random() < topo['mule_cluster_density']:
                if d == 0 and np.random.random() < mule_density:
                    
                    num_splits = int(topo['fan_out_ratio']) 
                    num_splits = int(f_out_ratio)  # DEFENSIVE: Ensures range() never crashes
                    
                    #split_amt = flow_amt / num_splits  # Divide the flow among branches
                    split_amt = amt / num_splits # Divide the hybrid amt among branches
                    
                    for _ in range(num_splits):
                        # Record each separate split as its own transaction
                        rows.append(_assemble_tx(
                            acct, party_key, ref_date, t_new, round(split_amt, 2), tx_type, cfg,
                            pools={"country_pool_primary": cfg['country_pool_primary'], "offshore_prob": 0.5},
                            label=current_label,
                        ))
                        # Track each split for the next layering hop
                        next_layer_funds.append({"amt": split_amt, "time": t_new})
                
                # B. FAN-IN: Consolidate funds toward the end (Only if not already branched)
                #elif d == n_hops - 1 and random.random() < topo['fan_in_ratio']:
                elif d == n_hops - 1 and np.random.random() < f_in_ratio:                    
                    # Note: Logic assumes you consolidate toward one target. 
                    # n_hops replaces the undefined n_depth.
                    rows.append(_assemble_tx(
                        acct, party_key, ref_date, t_new, round(amt, 2), tx_type, cfg,
                        pools={"country_pool_primary": cfg['country_pool_primary'], "offshore_prob": 0.5},
                        label=current_label,
                    ))
                    #next_layer_funds.append({"amt": flow_amt, "time": t_new})
                
                # C. STANDARD HOP: Regular 1-to-1 transfer
                else:
                    rows.append(_assemble_tx(
                        acct, party_key, ref_date, t_new, round(amt, 2), tx_type, cfg,
                        label=current_label,
                        pools={"country_pool_primary": cfg['country_pool_primary'], "offshore_prob": 0.5},
                    ))
                    next_layer_funds.append({"amt": amt, "time": t_new})
                    

            active_funds = next_layer_funds

        # --- 5. MICRO-TRANSACTIONS (Final Integration) ---
        #for f in active_funds:
            # 5. MICRO-TRANSACTIONS (Final Fan-out)
            #rows.extend(_generate_micro_tx(f['time'], f['amt'], cfg,
                                       #pools={"country_pool_primary": cfg['country_pool_primary'],
                                              #"offshore_pool": _HIGH_RISK, 
                                              #"domestic_pool": _ASIA + _MAJOR}, ref_date=ref_date, label=current_label ))
            
