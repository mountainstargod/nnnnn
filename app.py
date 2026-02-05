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
            
    
    # ------------------ Biz Scenarios ------------------
    

    elif scenario_type == "biz_inflow_outflow_ratio":
        # -----------------------------------------------
        # FETCH SCENARIO CONFIG
        # -----------------------------------------------
        cfg = scenario_config['biz_inflow_outflow_ratio']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']  

        # -----------------------------------------------
        # 2. SYNTHETIC DEMOGRAPHICS (SAMPLING)
        # -----------------------------------------------
        #age = np.random.uniform(20, 70)
        #occupation = np.random.choice(sampling_cfg['Occupation']['choices'], p=sampling_cfg['Occupation']['probabilities'])
        #industry = np.random.choice(sampling_cfg['Industry']['choices'], p=sampling_cfg['Industry']['probabilities'])
        #entity_type = np.random.choice(sampling_cfg['EntityType']['choices'], p=sampling_cfg['EntityType']['probabilities'])
        #risk_score = np.random.randint(sampling_cfg['RiskScore']['low'], sampling_cfg['RiskScore']['high'] + 1)
        #revenue_band = np.random.lognormal(mean=9, sigma=1)


        def dynamic_sample(field_name):
            """
            Categorical sampler for Business Inflow/Outflow.
            Navigates the ['probabilities'] nesting in sampling_cfg.
            """
            # 1. OPTUNA KEY: e.g., 'biz_inflow_outflow_ratio_Occupation_probs'
            prob_key = f"{scenario_type}_{field_name}_probs"
            
            # 2. BASELINE SOURCING: Pulls from sampling_cfg['Occupation']['probabilities']
            # This strips the scenario prefix to find the static prior
            field_cfg = sampling_cfg.get(field_name, {})
            default_probs = field_cfg.get('probabilities', [])
            
            # 3. OVERRIDE: Prioritize the Bayesian shifted vector from Optuna
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            # 4. SAMPLE: Returns the choice string (e.g., 'Consulting') directly
            choices = field_cfg.get('choices', [])
            return np.random.choice(choices, p=p_vector)
        
        # Execution in generate_events_from_params for biz_inflow_outflow_ratio
        actor = {
            # Categorical Features (Optimized via Block 0)
            "Occupation": dynamic_sample("Occupation"),
            "Industry":   dynamic_sample("Industry"),
            "EntityType": dynamic_sample("EntityType"),
            
            # Continuous/Integer Features (Optimized via Optuna Overrides)
            "Age":         opt_overrides.get(f"{scenario_type}_age", np.random.uniform(20, 70)),
            
            "RiskScore":   opt_overrides.get(f"{scenario_type}_risk_score", 
                                             np.random.randint(sampling_cfg['RiskScore']['low'], 
                                                               sampling_cfg['RiskScore']['high'] + 1)),
                                             
            "RevenueBand": opt_overrides.get(f"{scenario_type}_revenue_band", 
                                             np.random.lognormal(mean=9, sigma=1))
        }

        # --- 2026 SCOPE RECONCILIATION (The "Safety Unpack") ---
        # We pull these out to satisfy legacy downstream references to standalone variables
        age          = actor["Age"]
        occupation   = actor["Occupation"]
        industry     = actor["Industry"]
        entity_type  = actor["EntityType"]
        risk_score   = actor["RiskScore"]
        revenue_band = actor["RevenueBand"]


        
        # ------------------------------------------------------------
        # 3. DEMOGRAPHICS ? NUMERICAL EFFECTS (RE-ENABLED)
        # ------------------------------------------------------------
        # AGE: Numerical Calculation
        #age_f = multipliers['Age']
        #age_factor = age_f['base'] + (age_f['offset'] - age) / age_f['divisor']

        # OCCUPATION, INDUSTRY, ENTITY: String-to-Weight Lookups
        #occupation_risk   = multipliers['Occupation'].get(occupation, 1.0)
        #industry_factor   = multipliers['Industry'].get(industry, 1.0)
        #entity_multiplier = multipliers['EntityType'].get(entity_type, 1.0)
        
        # RISK & REVENUE: Direct Scaling
        #risk_prob_boost   = risk_score * multipliers['RiskScore_Scale']
        #rev_factor        = min(max(revenue_band / multipliers['Revenue_Scale'], 0.5), 5.0)


        # --- RECONCILED LOOKUPS (Bayesian Override vs. Config Fallback) ---
        
        # 1. AGE FACTOR: Optimize the base and offset
        # Logic: age_f['base'] + (age_f['offset'] - age) / age_f['divisor']
        a_base = opt_overrides.get(f"{scenario_type}_Age_base", multipliers['Age']['base'])
        a_off  = opt_overrides.get(f"{scenario_type}_Age_offset", multipliers['Age']['offset'])
        age_factor = a_base + (a_off - age) / multipliers['Age']['divisor']
    
        # 2. CATEGORICAL LOOKUPS: Occupation, Industry, Entity
        # Logic: Uses get_factor to handle Dict vs Float reconciliation
        occupation_risk = opt_overrides.get(
            f"{scenario_type}_Occ_f", 
            get_factor(multipliers.get('Occupation'), occupation)
        )
        industry_factor = opt_overrides.get(
            f"{scenario_type}_Industry_f", 
            get_factor(multipliers.get('Industry'), industry)
        )
        entity_multiplier = opt_overrides.get(
            f"{scenario_type}_Entity_f", 
            get_factor(multipliers.get('EntityType'), entity_type)
        )
    
        # 3. SCALING FACTORS: Risk and Revenue
        # Risk Logic: risk_score * multipliers['RiskScore_Scale']
        risk_f_scale = opt_overrides.get(f"{scenario_type}_Risk_f", multipliers['RiskScore_Scale'])
        risk_prob_boost = risk_score * risk_f_scale

        # --- RECONCILED REVENUE CLIPPING (Optimizing hardcoded boundaries) ---
        
        # 1. Fetch suggestions for boundaries (fallback to 0.5 and 5.0)
        rev_min_bound = opt_overrides.get(f"{scenario_type}_Rev_min", 0.5)
        rev_max_bound = opt_overrides.get(f"{scenario_type}_Rev_max", 5.0)
        
        # 2. Fetch suggestion for scaling denominator
        # Revenue Logic: min(max(revenue_band / multipliers['Revenue_Scale'], 0.5), 5.0)
        rev_f_scale = opt_overrides.get(f"{scenario_type}_Rev_f", multipliers['Revenue_Scale'])
        
        rev_factor = min(max(revenue_band / rev_f_scale, rev_min_bound), rev_max_bound)

        

        # Global Demographic Multiplier (D) - No longer 1!
        D = occupation_risk * industry_factor * entity_multiplier * rev_factor 

        # 1. Upfront 3D Parameter Extraction
        topo, chan, econ = cfg['network_topology'], cfg['channel_config'], cfg['economic_sensibility']
        
        # 2. 3D RISK MODIFIER (The Third Dimension)
        # 2026 Standard: Cluster Density, Rail Hopping, and Staged Escalation drive the Risk Mod
        #total_risk_mod = (topo['cluster_density'] * 1.5 if topo['cluster_density'] > 0.5 else 1.0) * \
                         #(chan['rail_hopping_intensity'] / 2.0) * \
                         #(econ['staged_escalation_factor'] if np.random.random() < 0.2 else 1.0)

        # Apply a non-zero floor to the entire product
        total_risk_mod = max(1e-6, 
            (topo['cluster_density'] * 1.5 if topo['cluster_density'] > 0.5 else 1.0) * \
            (chan['rail_hopping_intensity'] / 2.0) * \
            (econ['staged_escalation_factor'] if np.random.random() < 0.2 else 1.0)
        )

        # --- 2. THE 2026 UNIFIED BEHAVIORAL RISK (UBR) ---
        # Merges Static Risk (D_Risk) with Dynamic Risk (Topo, Chan, Econ)
        # We use the 'Total_Behavioral_Risk' architecture for cross-typology symmetry

        # --- 1. CONSOLIDATED DEMOGRAPHIC ACTOR (D_Risk) ---
        # We integrate the multipliers directly to create a "Static Risk Foundation"

        # --- 2026 PRODUCTION-GRADE BIO RISK BRIDGE ---
        # 1. Base Demographic Risk
        D_Risk = (occupation_risk * industry_factor * entity_multiplier * rev_factor * age_factor)

        # 2. Combined Behavioral Risk with 1e-6 Safeguard
        # This protects against 0-value multipliers suggested by Bayesian Optimization
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            (topo['cluster_density'] * 1.5 if topo['cluster_density'] > 0.5 else 1.0) *
            (chan['rail_hopping_intensity'] / 2.0) *
            (econ['staged_escalation_factor'])
        ))


        # --- RECONCILED BEHAVIORAL RISK (Block 3 Integration) ---
        
        # 1. Cluster Density: Optimize the 1.5 multiplier
        # Fallback remains 1.5 if Optuna hasn't reached this block yet
        c_base = opt_overrides.get(f"{scenario_type}_cluster_density_base", topo['cluster_density'])
        c_mult = opt_overrides.get(f"{scenario_type}_cluster_multiplier_f", 1.5)
        c_thresh = opt_overrides.get(f"{scenario_type}_cluster_threshold", 0.5)
        c_floor  = opt_overrides.get(f"{scenario_type}_cluster_floor", 1.0)        
        cluster_component = c_mult if c_base > c_thresh else c_floor
        
        # 2. Rail Hopping: Optimize the 2.0 divisor
        # Fallback remains 2.0
        r_base = opt_overrides.get(f"{scenario_type}_rail_hopping_base", chan['rail_hopping_intensity'])
        r_div = opt_overrides.get(f"{scenario_type}_rail_intensity_divisor", 2.0)
        rail_component = r_base / r_div
        
        # 3. Economic Escalation: Optimize the factor from config
        # Fallback is the raw config value
        e_f = opt_overrides.get(f"{scenario_type}_econ_escalation_f", econ['staged_escalation_factor'])
        
        # 4. Final Calculation with 1e-6 Safeguard
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            cluster_component *
            rail_component *
            e_f
        ))


        
        # 3. INTENSITY CALCULATION
        # Behavior is now a direct function of the structural risk modifier
        #intensity = np.random.uniform(2.0, 4.5) * total_risk_mod

        # Restore the baseline chance for high-intensity behavior
        base_intensity = np.random.uniform(2.0, 4.5) if np.random.random() < 0.25 else 1.0

        # --- RECONCILED GENERATIVE INTENSITY (Block 4) ---
        
        # 1. Behavioral Mode logic
        a_low  = opt_overrides.get(f"{scenario_type}_agg_low", 2.0)
        a_high = opt_overrides.get(f"{scenario_type}_agg_high", 4.5)
        a_prob = opt_overrides.get(f"{scenario_type}_agg_prob", 0.25)
        b_norm = opt_overrides.get(f"{scenario_type}_base_normal", 1.0) # Optimized Baseline
        
        # Logic: Randomly choose between Aggressive distribution or Optimized Normal baseline
        base_intensity = np.random.uniform(a_low, a_high) if np.random.random() < a_prob else b_norm

        
        intensity = base_intensity * total_risk_mod

        # 1. Determine base behavioral mode (Normal vs. Aggressive)
        #base_intensity = np.random.uniform(2.0, 4.5) if np.random.random() < 0.25 else 1.0
        
        # 2. Scale by the unified 2026 Risk Bridge
        # This ensures a 'Corporate PEP' in a 'High Risk Jurisdiction' 
        # moves money faster than a 'Low Risk Individual' even in 'Normal' mode.
        intensity = base_intensity * Total_Behavioral_Risk

        # --- 3. DYNAMIC INTENSITY & SAFEGUARDS ---
        # BO suggests these bounds to find the "Detection Phase Transition"
        int_floor = trial.suggest_float("bio_int_floor", 0.1, 1.0)
        int_ceiling = trial.suggest_float("bio_int_ceiling", 5.0, 15.0)

        
        # 3. Dynamic Safeguards & Precision
        int_floor   = opt_overrides.get(f"{scenario_type}_int_floor", 0.1)
        int_ceiling = opt_overrides.get(f"{scenario_type}_int_ceiling", 15.0)
        int_prec    = int(opt_overrides.get(f"{scenario_type}_int_precision", 2))

        
        # 3. Apply the BO-tuned Safeguard
        intensity = max(int_floor, min(int_ceiling, round(intensity, 2)))

        # 4. Final Intensity Clamping
        intensity = max(int_floor, min(int_ceiling, round(intensity, int_prec)))        
        
        # 4. REFINED LABELING LOGIC (3D Aware)
        # The threshold (2.0) is dynamically adjusted by the total_risk_mod
        #if (intensity > (2.0 / total_risk_mod)) or (risk_score >= 8 and intensity > 1.5) or (industry in ['cash_heavy'] and intensity > 1.2):
            #current_label = 1
        #else:
            #current_label = 0

        # Dynamic Risk Score for 2026 Labeling
        dynamic_risk = risk_score * (1 + (total_risk_mod - 1) * 0.5)
        if (intensity > 2.2) or (dynamic_risk >= 8 and intensity > 1.5) or (topo['circularity_prob'] > 0.7):
            current_label = 1
        else:
            current_label = 0

        # 1. Dynamic Risk Score (2026 Refinement)
        # Integrates the comprehensive behavioral risk into the base profile
        dynamic_risk = risk_score * (1 + (Total_Behavioral_Risk - 1) * 0.4)


        r_sens    = opt_overrides.get(f"{scenario_type}_risk_sensitivity_f", 0.4)
        r_offset  = opt_overrides.get(f"{scenario_type}_risk_base_offset", 1.0)
        r_neutral = opt_overrides.get(f"{scenario_type}_risk_neutral_floor", 1.0)
        
        # Refactored Formula: No hardcoded constants remain
        dynamic_risk = risk_score * (r_offset + (Total_Behavioral_Risk - r_neutral) * r_sens)
        
        
        # 2. 2026 Multi-Factor Labeling Logic
        # Suspicious behavior is now a fusion of Intensity, Profile Risk, and Topology
        label_conditions = [
            (intensity > 2.5),                                  # Pure Behavioral Surge
            (dynamic_risk >= 7.5 and intensity > 1.2),           # Risk-Proportional Anomaly
            (topo.get('circularity_prob', 0) > 0.65),           # Network Topology (Layering)
            (chan.get('rail_hopping_intensity', 0) > 3.0)        # Obfuscation Intensity (2026 Red Flag)
        ]


        # 2. Optimized Threshold Resolution
        # Resolving variables for the Decision Matrix
        p_int_thresh = opt_overrides.get(f"{scenario_type}_pure_intensity_thresh", 2.5)
        prop_risk_m  = opt_overrides.get(f"{scenario_type}_proportional_risk_min", 7.5)
        prop_int_m   = opt_overrides.get(f"{scenario_type}_proportional_int_min", 1.2)
        
        # Network/Channel Parameter Refactoring (Optimized vs Config)
        circ_val    = opt_overrides.get(f"{scenario_type}_circularity_prob_base", topo.get('circularity_prob', 0))
        circ_thresh = opt_overrides.get(f"{scenario_type}_circularity_thresh", 0.65)
        
        rail_val    = opt_overrides.get(f"{scenario_type}_rail_hopping_base", chan.get('rail_hopping_intensity', 0))
        rail_thresh = opt_overrides.get(f"{scenario_type}_rail_obfuscation_thresh", 3.0)
        
        # 3. Final Decision Matrix (Recall-Optimized)
        label_conditions = [
            (intensity > p_int_thresh),                      # Pure Behavioral Surge
            (dynamic_risk >= prop_risk_m and intensity > prop_int_m), # Risk-Proportional Anomaly
            (circ_val > circ_thresh),                        # Network Topology (Layering)
            (rail_val > rail_thresh)                         # Obfuscation Intensity (Red Flag)
        ]

        
        current_label = 1 if any(label_conditions) else 0

        
        
        # -----------------------------------------------
        # 4. TRANSACTION GENERATION (ECONOMIC SENSIBILITY)
        # -----------------------------------------------
        n_tx = max(5, int(cfg.get('n_tx_base') * intensity))
        #t_prev = ref_date - timedelta(days=np.random.uniform(25, 29))

        # 1. Transaction Count Optimization
        n_floor = int(opt_overrides.get(f"{scenario_type}_n_tx_floor", 5))
        n_base  = opt_overrides.get(f"{scenario_type}_n_tx_base_val", cfg.get('n_tx_base', 10)) # Assuming 10 as default
        n_tx    = max(n_floor, int(n_base * intensity))
        
        # --- 2026 REFINEMENT: MACRO-START OFFSET ---
        # High-risk entities start their "Inflow-Outflow" cycle with more urgency
        # We scale the 25-day buffer by the holistic risk
        #start_buffer_days = np.random.uniform(20, 30) / (intensity * max(0.5, total_risk_mod)) 
        start_buffer_days = np.random.uniform(20, 30) / (intensity * max(0.5, Total_Behavioral_Risk)) 

        # 2. Temporal Buffer Optimization (Corrected to Total_Behavioral_Risk)
        # Logic: np.random.uniform(20, 30) / (intensity * max(0.5, Total_Behavioral_Risk))
        b_low    = opt_overrides.get(f"{scenario_type}_buffer_days_low", 20.0)
        b_high   = opt_overrides.get(f"{scenario_type}_buffer_days_high", 30.0)
        beh_floor = opt_overrides.get(f"{scenario_type}_behavioral_risk_floor", 0.5)
        
        # Applying the optimized risk modifier floor to the 2026 Risk Bridge
        start_buffer_days = np.random.uniform(b_low, b_high) / (intensity * max(beh_floor, Total_Behavioral_Risk))


        t_prev = ref_date - timedelta(days=start_buffer_days)


        
        # --- PRE-LOOP: Define the Bayesian Strategy and Bounds ---
        
        # 1. Define the Bayesian Strategy once to ensure entity consistency
        # We define the 0.95 - 1.0 relationship via the Bayesian Shift
        # This ensures the 'Mean' of the distribution is 95% to 100% of the Inflow
        biz_m_shift = trial.suggest_float("biz_parity_shift", 0.95, 1.0)
        
        # We shrink the spread to create 'Mechanical Precision' (Low Entropy)
        biz_s_scale = trial.suggest_float("biz_parity_precision", 0.1, 0.4)


        # 3. Parity Logic (Replaces direct trial calls)
        biz_m_shift = opt_overrides.get(f"{scenario_type}_biz_parity_shift", 0.97) # Midpoint of 0.95-1.0
        biz_s_scale = opt_overrides.get(f"{scenario_type}_biz_parity_precision", 0.25) # Midpoint of 0.1-0.4


        # 2. Define the global system bounds
        lower_bound = cfg['amt_min']
        upper_bound = cfg['amt_max']

        # 4. Amount Boundary Optimization
        lower_bound = opt_overrides.get(f"{scenario_type}_amt_min_base", cfg.get('amt_min'))
        upper_bound = opt_overrides.get(f"{scenario_type}_amt_max_base", cfg.get('amt_max'))
        
        for _ in range(n_tx):
            # ECONOMIC SENSIBILITY: Revenue Velocity Cap (Ensures volume fits profile)
            # 2026 APAC Retail Seasonality (vol_idx) also scales the base amount
            seasonal_boost = 1.0 + (econ['seasonal_volatility_index'] if industry == 'retail' else 0.0)

            # 1. Economic Sensibility: Seasonality & Revenue Velocity
            s_base  = opt_overrides.get(f"{scenario_type}_seasonal_boost_base", 1.0)
            s_vol   = opt_overrides.get(f"{scenario_type}_seasonal_vol_idx", econ['seasonal_volatility_index'])
            
            # 2026 logic: Apply boost only if industry matches 'retail'
            seasonal_boost = s_base + (s_vol if industry == 'retail' else 0.0)

            
            base_amt = np.random.uniform(cfg['amt_min'], cfg['amt_max']) * D * seasonal_boost
            base_amt = min(base_amt, revenue_band * econ['revenue_to_velocity_cap'] / n_tx)

            
            # Amount calculation using optimized boundaries from Block 6 and seasonal boost from Block 7
            # lower_bound/upper_bound/D are from previous blocks
            base_amt = np.random.uniform(lower_bound, upper_bound) * D * seasonal_boost
            v_cap_f = opt_overrides.get(f"{scenario_type}_rev_velocity_cap_f", econ['revenue_to_velocity_cap'])
            
            # Applying the Velocity Cap (Revenue Band * Cap Factor / Transaction Count)
            base_amt = min(base_amt, (revenue_band * v_cap_f) / n_tx)

            

            
            # CHANNEL DIVERSITY: Rail Hopping logic
            is_rail_hop = np.random.random() < (0.3 * chan['rail_hopping_intensity'])
            #tx_type = np.random.choice(chan['rails']) if is_rail_hop else "TRANSFER"

            # 2. Channel Diversity: Rail Hopping logic
            # Logic: np.random.random() < (0.3 * intensity)
            rh_static = opt_overrides.get(f"{scenario_type}_rail_hop_prob_static", 0.3)
            rh_intens = opt_overrides.get(f"{scenario_type}_rail_hop_intensity_f", chan['rail_hopping_intensity'])
            
            is_rail_hop = np.random.random() < (rh_static * rh_intens)


            # --- 2026 HYBRID CHANNEL/RATIO LOGIC ---
            if is_rail_hop:
                # Use a specific high-risk rail for both sides
                selected_rail = np.random.choice(chan['rails'])
                in_tx_type = f"CREDIT_{selected_rail}"
                out_tx_type = f"DEBIT_{selected_rail}"
            else:
                # Fallback to generic TRANSFER if not hopping rails
                in_tx_type = "CREDIT_TRANSFER"
                out_tx_type = "DEBIT_TRANSFER"
            
            # Temporal velocity bypass for instant rails
            v_bypass = chan['settlement_latency_bypass'] if is_rail_hop else 1.0

            # 1. Settlement Latency Bypass logic
            # Logic: Use bypass if rail hopping, else fallback to baseline
            lat_bypass = opt_overrides.get(f"{scenario_type}_latency_bypass_base", chan['settlement_latency_bypass'])
            lat_norm   = opt_overrides.get(f"{scenario_type}_latency_bypass_fallback", 1.0)
            
            v_bypass = lat_bypass if is_rail_hop else lat_norm



            #t_prev += timedelta(hours=float(cfg['tx_gap_hours'] * v_bypass * intensity))
            #t_prev += timedelta(hours=float(max(1e-6, cfg['tx_gap_hours'] * v_bypass * intensity)))

            
            # High intensity and v_bypass should REDUCE the gap (increase velocity)
            # Restoring the exponential distribution for realistic jitter
            #gap_scale = cfg['tx_gap_hours'] / (intensity * v_bypass)

            # --- 2026 REFINEMENT: HOLISTIC COMPRESSION ---
            # We ensure BOTH intensity and Total_Behavioral_Risk drive the velocity   # total_risk_mod
            # This creates the "Criminal Signature" needed for high F1 uplift
            #holistic_velocity = intensity * v_bypass * max(0.5, total_risk_mod)
            holistic_velocity = intensity * v_bypass * max(0.5, Total_Behavioral_Risk)


            # 2. Holistic Velocity Scaling
            # Incorporates the optimized intensity and the behavior-based risk floor
            v_floor = opt_overrides.get(f"{scenario_type}_velocity_risk_floor", 0.5)
            holistic_velocity = intensity * v_bypass * max(v_floor, Total_Behavioral_Risk)

            gap_scale = cfg['tx_gap_hours'] / holistic_velocity

            # 3. Temporal Gap Calculation
            gap_base  = opt_overrides.get(f"{scenario_type}_tx_gap_hours_base", cfg['tx_gap_hours'])
            gap_scale = gap_base / holistic_velocity


            # Update t_prev with risk-weighted exponential gap
            t_prev += timedelta(hours=float(np.random.exponential(scale=gap_scale)))

            
            # Re-insert jitter to avoid synthetic fingerprints
            #t_prev += timedelta(minutes=np.random.randint(-30, 30))

            # Jitter to avoid fingerprints
            t_prev += timedelta(minutes=np.random.randint(-15, 15))


            # 4. Temporal Jitter Optimization
            # Replaces hardcoded -15 to 15 range
            j_min = int(opt_overrides.get(f"{scenario_type}_jitter_min", -15))
            j_max = int(opt_overrides.get(f"{scenario_type}_jitter_max", 15))
            
            t_prev += timedelta(minutes=np.random.randint(j_min, j_max + 1))

            # Determine Inflow vs Outflow
            anomalous_prob = 0.5
            if np.random.rand() < anomalous_prob:
                in_amt = base_amt * (cfg['inflow_multiplier_high'] ** intensity) * occupation_risk
                out_amt = base_amt * (cfg['outflow_multiplier_low'] ** intensity) * industry_factor
            else:
                in_amt = base_amt * (cfg['inflow_multiplier_low'] ** intensity) * entity_multiplier
                out_amt = base_amt * (cfg['outflow_multiplier_high'] ** intensity) * age_factor


            # --- RECONCILED DUAL-LAYER MAGNITUDE SCALING (Block 8) ---
            
            # 1. Resolve Global Scaling Weights
            a_prob   = opt_overrides.get(f"{scenario_type}_anomalous_prob_thresh", 0.5)
            b_weight = opt_overrides.get(f"{scenario_type}_behavioral_weight_f", 1.0)
            d_weight = opt_overrides.get(f"{scenario_type}_demographic_weight_f", 1.0)
            
            # The "Suspicion Scalar" (2026 Bridge) drives the Anomaly Magnitude
            suspicion_scalar = Total_Behavioral_Risk * b_weight
            
            # 2. Optimized Multipliers
            i_high = opt_overrides.get(f"{scenario_type}_inflow_mult_high", cfg['inflow_multiplier_high'])
            i_low  = opt_overrides.get(f"{scenario_type}_inflow_mult_low", cfg['inflow_multiplier_low'])
            o_high = opt_overrides.get(f"{scenario_type}_outflow_mult_high", cfg['outflow_multiplier_high'])
            o_low  = opt_overrides.get(f"{scenario_type}_outflow_mult_low", cfg['outflow_multiplier_low'])
            
            # 3. Dual-Layer Amount Generation
            # Logic: Base * (Typology Growth) * (Weighted Demographic Anchor) * (Global Suspicion)
            if np.random.rand() < a_prob:
                # Mode A: Occupation-anchored Inflow vs Industry-anchored Outflow
                in_amt  = base_amt * (i_high ** intensity) * (occupation_risk * d_weight) * suspicion_scalar
                out_amt = base_amt * (o_low ** intensity)  * (industry_factor * d_weight) * suspicion_scalar
            else:
                # Mode B: Entity-anchored Inflow vs Age-anchored Outflow
                in_amt  = base_amt * (i_low ** intensity)  * (entity_multiplier * d_weight) * suspicion_scalar
                out_amt = base_amt * (o_high ** intensity) * (age_factor * d_weight) * suspicion_scalar


        # --- THE PYTHON IMPLEMENTATION OF MEAN TRANSLATION AND SIGMA CONTRACTION ---
        
            # 1. Translate the Mean Upwards 
            # This implements your 0.95-1.0 requirement mathematically.
            # If in_amt is 1000 and biz_m_shift is 0.97, the center of our distribution is 970.
            tuned_mu = in_amt * biz_m_shift
            

            # 2. Shrink the Standard Deviation
            # Start with a base variance (e.g., 2% of the amount) and contract it by our strategy (biz_s_scale)
            tuned_sigma = (in_amt * 0.02) * biz_s_scale 


            # Resolving the 0.02 and 0.01 hardcoded figures
            s_coeff = opt_overrides.get(f"{scenario_type}_sigma_base_coeff", 0.02)
            
            # Logic: (Amount * Optimized Coeff) * Optimized Scale
            tuned_sigma = (in_amt * s_coeff) * biz_s_scale


            # Ensure sigma is never zero to prevent division errors
            safe_sigma = max(0.01, tuned_sigma)

            s_floor = opt_overrides.get(f"{scenario_type}_sigma_floor", 0.01)
            safe_sigma  = max(s_floor, tuned_sigma)
        
            # 3. Calculate Truncation Bounds (Z-scores)
            # This maps the global [min, max] constraints to our tuned distribution
            a_lay = (lower_bound - tuned_mu) / safe_sigma
            b_lay = (upper_bound - tuned_mu) / safe_sigma

            # 3. Optimized Circularity Rounding
            c_prec = int(opt_overrides.get(f"{scenario_type}_circularity_round_prec", 2))
            
            # ECONOMIC SENSIBILITY: Balance Retention Floor (Mule residue)
            # NETWORK TOPOLOGY: Circularity check (U-Turn logic)
            #if np.random.random() < topo['circularity_prob'] and current_label == 1:

            # circ_val was resolved in Block 5
            if np.random.random() < circ_val and current_label == 1:
                
                # Force U-Turn: Inflow and Outflow match within 5% to simulate circularity
                #out_amt = in_amt * np.random.uniform(0.95, 1.0)
                # 4. Generate the Amount using Truncated Normal Distribution
                # This generates an 'out_amt' that is centered on 'tuned_mu' with 'tuned_sigma' spread
                #out_amt = in_amt * round(truncnorm.rvs(a_lay, b_lay, loc=tuned_mu, scale=safe_sigma)
                
                #out_amt = round(truncnorm.rvs(a_lay, b_lay, loc=tuned_mu, scale=safe_sigma), 2)

                out_amt = round(truncnorm.rvs(a_lay, b_lay, loc=tuned_mu, scale=safe_sigma), c_prec)

                        
            # Retention logic (Money must "legitimately" rest in the account)
            #retained_amt = out_amt * econ['balance_retention_floor']
            #final_out_amt = out_amt - retained_amt

            # If circularity is detected (laundering), retention is near-zero
            effective_retention = 0.01 if (current_label == 1 and np.random.random() < topo['circularity_prob']) else econ['balance_retention_floor']

            # 1. Optimized Retention (Leakage) Calculation
            # Logic: If circular layering, use optimized low-leakage factor; else use config floor
            c_ret_f = opt_overrides.get(f"{scenario_type}_circular_retention_f", 0.01)
            r_floor = opt_overrides.get(f"{scenario_type}_retention_floor_base", econ['balance_retention_floor'])
            
            # circ_val (circularity_prob_base) from Block 5
            is_circular = (current_label == 1 and np.random.random() < circ_val)
            effective_retention = c_ret_f if is_circular else r_floor


            final_out_amt = out_amt * (1 - effective_retention)

            # Record Inflow
            #rows.append(_assemble_tx(acct, party_key, ref_date, t_prev, round(in_amt, 2), "CREDIT", cfg, label=current_label))
            rows.append(_assemble_tx(acct, party_key, ref_date, t_prev, round(in_amt, 2), in_tx_type, cfg, label=current_label))
            
            # Record Outflow (Timing controlled by U-Turn threshold)
            t_out = t_prev + timedelta(hours=np.random.uniform(2, topo['u_turn_threshold_hours']))

            # 3. Optimized U-Turn Temporal Spacing
            # Logic: timedelta(hours=np.random.uniform(2, threshold))
            u_min    = opt_overrides.get(f"{scenario_type}_uturn_gap_min", 2.0)
            u_max_b  = opt_overrides.get(f"{scenario_type}_uturn_gap_max_base", topo['u_turn_threshold_hours'])            
            # t_out determines when the funds leave after arriving
            t_out = t_prev + timedelta(hours=np.random.uniform(u_min, u_max_b)


            #rows.append(_assemble_tx(acct, party_key, ref_date, t_out, round(final_out_amt, 2), "DEBIT", cfg, label=current_label))
            rows.append(_assemble_tx(acct, party_key, ref_date, t_out, round(final_out_amt, 2), out_tx_type, cfg, label=current_label))


    elif scenario_type == "biz_monthly_volume_deviation":
        # -----------------------------------------------
        # FETCH SCENARIO CONFIG
        # -----------------------------------------------
        cfg = scenario_config['biz_monthly_volume_deviation']
        sampling_cfg = cfg['demographic_sampling']
        multipliers = cfg['demographic_factors']

        # Extract 2026 3D Parameters Upfront
        topo, chan, econ = cfg['network_topology'], cfg['channel_config'], cfg['economic_sensibility']        

        # Helper function to handle flattened dictionary from Bayesian Optimization
        def get_factor(source, actor_val):
            # If it's the expected dict, do a lookup
            if isinstance(source, dict):
                return source.get(str(actor_val), 1.0)
            # If it's a flattened number (like your debug showed), use it directly
            if isinstance(source, (int, float)):
                return float(source)
            return 1.0
        
        # -----------------------------------------------
        # SYNTHETIC DEMOGRAPHICS
        # -----------------------------------------------
        actor = {
            "Age": np.random.choice(
                sampling_cfg['Age']['choices'],
                p=sampling_cfg['Age']['probabilities']
            ),
            "Region": np.random.choice(
                sampling_cfg['Region']['choices'],
                p=sampling_cfg['Region']['probabilities']
            ),
            "Occupation": np.random.choice(
                sampling_cfg['Occupation']['choices'],
                p=sampling_cfg['Occupation']['probabilities']
            ),
            "RiskScore": np.random.beta(
                a=sampling_cfg['RiskScore']['a'],
                b=sampling_cfg['RiskScore']['b']
            ) * 10,
            "BusinessSize": np.random.choice(
                sampling_cfg['BusinessSize']['choices'],
                p=sampling_cfg['BusinessSize']['probabilities']
            ),
            "Sophistication": np.random.choice(
                sampling_cfg['Sophistication']['choices'],
                p=sampling_cfg['Sophistication']['probabilities']
            )
        }


        # --- RECONCILED DYNAMIC SAMPLING (Block 0 Optimization) ---
        def dynamic_sample(field_name):
            """
            Fetches and samples categorical distributions for Monthly Volume Deviation.
            Prioritizes Optuna's Bayesian-shifted vectors over static config priors.
            """
            # 1. Resolve Optuna Key (e.g., 'biz_monthly_volume_deviation_Occupation_probs')
            prob_key = f"{scenario_type}_{field_name}_probs"
            
            # 2. Sourcing Baseline from sampling_cfg
            field_data = sampling_cfg.get(field_name, {})
            choices = field_data.get('choices', [])
            default_probs = field_data.get('probabilities', [])
            
            # 3. Bayesian Override: Prioritize the shifted vector from Optuna
            p_vector = opt_overrides.get(prob_key, default_probs)
            
            # 4. Safe Sampling: Ensure choice count matches probability vector length
            return np.random.choice(choices, p=p_vector)

        # --- 2026 RECONCILED ACTOR GENERATION ---
        actor = {
            # Categorical Features (Optimized via Block 0 Probability Shifts)
            "Age":            dynamic_sample("Age"),
            "Region":         dynamic_sample("Region"),
            "Occupation":     dynamic_sample("Occupation"),
            "BusinessSize":   dynamic_sample("BusinessSize"),
            "Sophistication": dynamic_sample("Sophistication"),

            # Continuous Features (Optimized via Singular Value Overrides)
            # Logic: Prioritize Optuna's suggested score, fallback to Beta distribution
            "RiskScore": opt_overrides.get(
                f"{scenario_type}_risk_score", 
                np.random.beta(a=sampling_cfg['RiskScore']['a'], b=sampling_cfg['RiskScore']['b']) * 10
            )
        }

        # --- SAFETY UNPACK (For downstream reference) ---
        age            = actor["Age"]
        region         = actor["Region"]
        occupation     = actor["Occupation"]
        risk_score     = actor["RiskScore"]
        business_size  = actor["BusinessSize"]
        sophistication = actor["Sophistication"]



        # --- 2. 3D TOTAL RISK MODIFIER ---
        # Fuses 2026 Burst Topology, Rail Hopping, and Staged Escalation
        total_risk_mod = max(1e-6,         
                         (topo['burst_node_centrality'] if actor['RiskScore'] > 6 else 1.0) * \
                         (chan['rail_switch_intensity'] / 2.0) * \
                         (econ['staged_escalation_factor'] if actor['Region'] == "HighRiskOffshore" else 1.0)
        )
        # 30% Suspicious Volume Deviation scaled by 3D structural risk
        if np.random.random() < 0.30:
            intensity = np.random.uniform(2.0, 4.5) * total_risk_mod
        else:
            intensity = 1.0

        # --- 1. COMPOUND DEMOGRAPHIC MULTIPLIERS ---
        # Map string-based choices to BO-tunable multipliers
        age_f      = get_factor(multipliers.get('Age'), actor['Age'])
        region_f   = get_factor(multipliers.get('Region'), actor['Region'])
        occ_f      = get_factor(multipliers.get('Occupation'), actor['Occupation'])
        biz_size_f = get_factor(multipliers.get('BusinessSize'), actor['BusinessSize'])
        soph_f     = get_factor(multipliers.get('Sophistication'), actor['Sophistication'])


        # --- RECONCILED DEMOGRAPHIC MULTIPLIERS (Block 1 Optimization) ---
        # 1. We prioritize the Optuna suggestion (singular float per trial)
        # 2. We fall back to the get_factor dictionary lookup from config
        
        age_f      = opt_overrides.get(f"{scenario_type}_Age_f", 
                                       get_factor(multipliers.get('Age'), actor['Age']))
                                       
        region_f   = opt_overrides.get(f"{scenario_type}_Region_f", 
                                       get_factor(multipliers.get('Region'), actor['Region']))
                                       
        occ_f      = opt_overrides.get(f"{scenario_type}_Occ_f", 
                                       get_factor(multipliers.get('Occupation'), actor['Occupation']))
                                       
        biz_size_f = opt_overrides.get(f"{scenario_type}_BizSize_f", 
                                       get_factor(multipliers.get('BusinessSize'), actor['BusinessSize']))
                                       
        soph_f     = opt_overrides.get(f"{scenario_type}_Soph_f", 
                                       get_factor(multipliers.get('Sophistication'), actor['Sophistication']))
        
        # Scaling the risk_score contribution
        # 1. Fetch the optimized multiplier (Defaults to 1.0 for the baseline trial)
        r_mult = opt_overrides.get(f"{scenario_type}_Risk_f", 1.0)
        
        # 2. Apply it to the RiskScore (which was optimized in Block 0)
        # This determines how heavily the RiskScore weights the final D_Risk
        risk_f = risk_score * r_mult


        # Total Demographic Foundation (D_Risk)
        # Note: RiskScore is already a float (0-10), so we use it as a direct scalar
        D_Risk = (age_f * region_f * occ_f * biz_size_f * soph_f * (actor['RiskScore'] / 5.0))

        # --- RECONCILED D_RISK (Block 1) ---
        # Logic: (age_f * region_f * occ_f * biz_size_f * soph_f * (actor['RiskScore'] / 5.0))
        # risk_f = (actor['RiskScore'] * r_mult) from previous step
        r_div = opt_overrides.get(f"{scenario_type}_risk_score_divisor", 5.0)
        
        D_Risk = (age_f * region_f * occ_f * biz_size_f * soph_f * (risk_f / r_div))


        # --- 2. 2026 UNIFIED BEHAVIORAL RISK (UBR) ---
        # We replace total_risk_mod with the Holistic Bridge
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            (topo['burst_node_centrality'] if actor['RiskScore'] > 6 else 1.0) * 
            (chan['rail_switch_intensity'] / 2.0) * 
            (econ['staged_escalation_factor'])
        ))


        # --- RECONCILED UNIFIED BEHAVIORAL RISK (UBR) ---
        
        # 1. Burst Node Centrality Optimization
        # Logic: (topo['burst_node_centrality'] if risk_score > 6 else 1.0)
        b_thresh = opt_overrides.get(f"{scenario_type}_burst_threshold", 6.0)
        c_f      = opt_overrides.get(f"{scenario_type}_centrality_f", topo['burst_node_centrality'])
        c_floor = opt_overrides.get(f"{scenario_type}_centrality_floor", 1.0)
        centrality_component = c_f if risk_score > b_thresh else c_floor

        # 2. Rail Switch Intensity Optimization
        # Logic: (chan['rail_switch_intensity'] / 2.0)
        rs_base = opt_overrides.get(f"{scenario_type}_rail_switch_base", chan['rail_switch_intensity'])
        rs_div = opt_overrides.get(f"{scenario_type}_rail_switch_divisor", 2.0)
        rail_component = rs_base / rs_div

        # 3. Staged Escalation Factor Optimization
        e_f = opt_overrides.get(f"{scenario_type}_econ_staged_f", econ['staged_escalation_factor'])

        # Final 2026 Unified Bridge with 1e-6 Safeguard
        Total_Behavioral_Risk = max(1e-6, (
            D_Risk * 
            centrality_component * 
            rail_component * 
            e_f
        ))



        # --- 3. DYNAMIC INTENSITY & BO SAFEGUARDS ---
        # BO suggests bounds to find the "Anomaly Detection Threshold"
        int_floor = trial.suggest_float("vol_int_floor", 0.1, 1.0)
        int_ceiling = trial.suggest_float("vol_int_ceiling", 5.0, 15.0)

        # 1. Superseded direct trial calls (Maintaining variable names for efficiency)
        int_floor = opt_overrides.get(f"{scenario_type}_vol_int_floor", 0.1)
        int_ceiling = opt_overrides.get(f"{scenario_type}_vol_int_ceiling", 15.0)


        # 1. Establish the 'Base Intent' for the month
        # 30% chance of an illicit 'Volume Burst' (Laundering/Capital Flight)
        # 70% chance of 'Business-as-Usual' (Normal operational variance)
        base_intensity = np.random.uniform(2.0, 4.5) if np.random.random() < 0.30 else 1.0



        # 1. Stochastic Mode Optimization
        # Logic: np.random.uniform(2.0, 4.5) if np.random.random() < 0.30 else 1.0
        a_low  = opt_overrides.get(f"{scenario_type}_agg_low", 2.0)
        a_high = opt_overrides.get(f"{scenario_type}_agg_high", 4.5)
        a_prob = opt_overrides.get(f"{scenario_type}_agg_prob", 0.30)
        b_norm = opt_overrides.get(f"{scenario_type}_base_normal", 1.0)

        base_intensity = np.random.uniform(a_low, a_high) if np.random.random() < a_prob else b_norm

        # 2. Bridge Intent to Profile (The 2026 'Contextual' Shift)
        # Replacing total_risk_mod with Total_Behavioral_Risk ensures 
        # a High-Risk Region SME has a higher intensity than a Low-Risk Individual
        # even if they both fall into the 'Base Intensity = 1.0' bucket.
        intensity = base_intensity * Total_Behavioral_Risk

        # 3. Apply the BO-tuned Dynamic Safeguard
        # This keeps the 'Volume Deviation' within plausible 2026 bounds
        intensity = max(int_floor, min(int_ceiling, round(intensity, 2)))


        
        # -----------------------------------------------
        # 3. ACTIVE MULTIPLIER LOOKUPS (RE-ENABLED)
        # -----------------------------------------------
        # Lookup the numerical weight for BusinessSize and Sophistication
        size_mult = multipliers['BusinessSize'].get(actor['BusinessSize'], 1.0)
        soph_mult = multipliers['Sophistication'].get(actor['Sophistication'], 1.0)

        # PROPOSED 2026 ACTIVATION:
        #region_mult = multipliers['Region'].get(actor['Region'], 1.0)
        #occ_mult = multipliers['Occupation_Risk'].get(actor['Occupation'], 1.0)

        # 1. Fetch the newly added Region
        region_mult = multipliers['Region'].get(actor['Region'], 1.0)
        
        # 2. Fix the naming mismatch
        # 3. Add [0] to extract the first value from the list in your config
        occ_mult = multipliers['Occupation'].get(actor['Occupation'], [1.0])[0]



        
        # Risk Score impacts frequency directly
        risk_weight = multipliers['RiskScore_Weight']
        actor_freq_multiplier = 1 + (risk_weight * actor['RiskScore'])
       
        # n_tx influenced by structural centrality
        #n_tx = max(5, int(cfg['n_tx_base'] * intensity * actor_freq_multiplier * (1 + topo['mule_cluster_density'])))

        # Use intensity for the flow volume, and topology density to slightly shift the baseline
        # This prevents the 'n_tx' from exploding while still reflecting a high-activity network
        n_tx_baseline = cfg['n_tx_base'] * actor_freq_multiplier * occ_mult * region_mult
        n_tx = max(5, int(n_tx_baseline * (intensity + topo['mule_cluster_density'])))


        # --- RECONCILED FREQUENCY LOGIC (Block 3) ---
        
        # 1. Resolve Parameters (Optimizing hardcoded 5, cfg['n_tx_base'], and topo['mule_cluster_density'])
        n_base = opt_overrides.get(f"{scenario_type}_n_tx_base_val", cfg['n_tx_base'])
        n_floor = int(opt_overrides.get(f"{scenario_type}_n_tx_floor", 5))
        m_density = opt_overrides.get(f"{scenario_type}_mule_density_base", topo['mule_cluster_density'])
        
        # 2. Optimized Frequency Scalar
        # Logic: Instead of re-calculating risk, we use the already-optimized intensity bridge.
        # We add a sensitivity factor to allow Optuna to tune the 'Volume Impact'
        f_sens = opt_overrides.get(f"{scenario_type}_freq_sensitivity_f", 1.0)
        
        # 3. Final n_tx Calculation (Consolidated)
        # This integrates intensity (Behavioral) and m_density (Network) 
        n_tx_baseline = n_base * Total_Behavioral_Risk * f_sens
        n_tx = max(n_floor, int(n_tx_baseline * (intensity + m_density)))




        # -----------------------------------------------
        # 3. DETERMINE MONTHLY DEVIATION FACTOR (ECONOMIC SENSIBILITY)
        # -----------------------------------------------
        # Economic Sensibility: High sophistication actors use cleaner (decayed) deviations
        s_decay = 1.0 - (econ['sophistication_decay'] if actor['Sophistication'] == 'High' else 0.0)

        # Combine Occupation and Region multipliers into the deviation magnitude
        # This ensures high-risk sectors/regions show larger financial spikes
        demographic_vol_mod = occ_mult * region_mult * size_mult
        
        if np.random.rand() < 0.5:
            high_low_range = cfg['monthly_deviation_alt_high'], cfg['monthly_deviation_high']
            monthly_deviation_factor = np.random.uniform(*high_low_range) * actor_freq_multiplier * demographic_vol_mod * s_decay
        else:
            low_range = cfg['monthly_deviation_low'], cfg['monthly_deviation_alt_low']
            monthly_deviation_factor = (np.random.uniform(*low_range) * (1 / actor_freq_multiplier) * demographic_vol_mod)

        # Economic Sensibility: Deviation Cap (Prevents profile-defying spikes)
        monthly_deviation_factor = min(monthly_deviation_factor, econ['revenue_deviation_cap'])


        # -----------------------------------------------
        # 3. REFINED MONTHLY DEVIATION (ECONOMIC SENSIBILITY)
        # -----------------------------------------------
        # Sophistication Decay: High sophistication = cleaner, less 'spiky' patterns
        s_decay = 1.0 - (econ['sophistication_decay'] if actor['Sophistication'] == 'High' else 0.0)


        # --- RECONCILED DEVIATION LOGIC (Block 3 Expansion) ---
        
        # 1. Sophistication Decay Optimization
        # Logic: 1.0 - (econ['sophistication_decay'] if actor['Sophistication'] == 'High' else 0.0)
        s_base  = opt_overrides.get(f"{scenario_type}_soph_decay_base", 1.0)
        s_idx   = opt_overrides.get(f"{scenario_type}_soph_decay_f", econ['sophistication_decay'])
        s_neut  = opt_overrides.get(f"{scenario_type}_soph_decay_neutral", 0.0)

        s_decay = s_base - (s_idx if actor['Sophistication'] == 'High' else s_neut)



        # 2026 REFINEMENT: Use Intensity as the primary driver of Magnitude
        # This links the 'how fast' (intensity) to the 'how much' (deviation)
        if np.random.rand() < 0.5:
            # HIGH DEVIATION (Positive Spike)
            # We use intensity to scale the uniform range
            high_low_range = cfg['monthly_deviation_alt_high'], cfg['monthly_deviation_high']
            monthly_deviation_factor = np.random.uniform(*high_low_range) * intensity * s_decay
        else:
            # LOW DEVIATION (Contraction/Baseline)
            # Intensity 1.0 = normal business. High intensity = suspicious contraction.
            low_range = cfg['monthly_deviation_low'], cfg['monthly_deviation_alt_low']
            monthly_deviation_factor = np.random.uniform(*low_range) * (1 / max(1.0, intensity))


        # 2. Deviation Range Optimization (High vs Low Modes)
        m_prob  = opt_overrides.get(f"{scenario_type}_mode_prob_thresh", 0.5)
        m_floor = opt_overrides.get(f"{scenario_type}_mode_safety_floor", 1.0)

        # 2. Risk-Magnitude Bridge
        # d_weight allows Optuna to decouple 'Magnitude' from 'Frequency' (intensity)
        d_weight = opt_overrides.get(f"{scenario_type}_dev_risk_weight_f", 1.0)
        effective_risk_scalar = Total_Behavioral_Risk * d_weight


        if np.random.rand() < m_prob:
            # Mode A: High Deviation
            d_alt_h = opt_overrides.get(f"{scenario_type}_dev_alt_high", cfg['monthly_deviation_alt_high'])
            d_h     = opt_overrides.get(f"{scenario_type}_dev_high", cfg['monthly_deviation_high'])
            monthly_deviation_factor = np.random.uniform(d_alt_h, d_h) * intensity * s_decay * effective_risk_scalar
        else:
            # Mode B: Low Deviation
            d_l     = opt_overrides.get(f"{scenario_type}_dev_low", cfg['monthly_deviation_low'])
            d_alt_l = opt_overrides.get(f"{scenario_type}_dev_alt_low", cfg['monthly_deviation_alt_low'])
            # intensity acts as an inverse scalar for low deviation mode
            monthly_deviation_factor = np.random.uniform(d_l, d_alt_l) * (effective_risk_scalar / max(m_floor, intensity))



        # 2026 ECONOMIC CAP:
        # Instead of a hard cap, use a dynamic cap based on BusinessSize
        # A large business can plausibly deviate more in absolute terms
        dynamic_cap = econ['revenue_deviation_cap'] * (2.0 if actor['BusinessSize'] == 'Large' else 1.0)
        monthly_deviation_factor = min(monthly_deviation_factor, dynamic_cap)



        # --- RECONCILED REVENUE DEVIATION CAP (Optimized for all sizes) ---
        
        # Resolve Base Cap
        cap_base  = opt_overrides.get(f"{scenario_type}_rev_dev_cap_base", econ['revenue_deviation_cap'])
        
        # Get the generalized factor. If Optuna doesn't override it, use the config's size map.
        # This assumes econ['cap_factors_by_size'] is a dictionary like {'Small': 1.0, 'Medium': 1.5, 'Large': 2.0}
        biz_size_cap_mult = opt_overrides.get(
            f"{scenario_type}_cap_size_f", 
            get_factor(econ.get('cap_factors_by_size'), actor['BusinessSize'])

        # Dynamic Cap Calculation: Uses the factor appropriate for the specific BusinessSize actor
        dynamic_cap = cap_base * biz_size_cap_mult
        
        # Apply the Optimized Cap
        monthly_deviation_factor = min(monthly_deviation_factor, dynamic_cap)

        
        # --- 4. REFINED LABELING LOGIC (3D AWARE) ---
        # 2026 Detection: Label is triggered by intensity OR structural depth (shell_depth_offset)
        #if (intensity > (2.2 / total_risk_mod)) or (monthly_deviation_factor > 2.5 and actor['RiskScore'] > 7.0) or (topo['shell_depth_offset'] > 2):
            #current_label = 1 
        #else:
            #current_label = 0 

        # --- REFINED LABELING LOGIC (RBA COMPLIANT 2026) ---
        
        # 1. Calibrate the Statistical Threshold
        # Since deviation now includes multipliers, we normalize the trigger
        # This prevents 'High Risk' actors from being flagged for baseline behavior
        effective_anomaly_threshold = 2.5 * (occ_mult * region_mult)
        

        # --- REFINED LABELING LOGIC (RBA COMPLIANT 2026) ---
        # 1. Integrate structural risk into a Dynamic Risk Score (DRS)
        # This keeps the threshold (2.2) stable while adjusting the actor's 'perceived' risk
        dynamic_risk = actor['RiskScore'] * (1 + (total_risk_mod - 1) * 0.5) 
        
        # 2. Multi-pronged detection: Behavioral, Statistical, and Structural

        # 3. Detection Triggers
        is_high_intensity = (intensity > 2.2) # Pure behavioral burst
        #is_stat_anomaly = (monthly_deviation_factor > 2.5 and dynamic_risk > 7.0)
        is_stat_anomaly = (monthly_deviation_factor > effective_anomaly_threshold and dynamic_risk > 7.0)
        is_structural_red_flag = (topo['shell_depth_offset'] > 2) # Network-based risk
        
        if is_high_intensity or is_stat_anomaly or is_structural_red_flag:
            current_label = 1 # Suspicious: High-confidence risk trigger
        else:
            current_label = 0 # Legitimate: Activity remains within risk-adjusted bound            





        # --- 1. DYNAMIC RISK SCORE (DRS) 2026 ---
        # We replace total_risk_mod with Total_Behavioral_Risk
        # This ensures the 'perceived' risk includes demographics + network behavior
        dynamic_risk = actor['RiskScore'] * (1 + (Total_Behavioral_Risk - 1) * 0.4) 


        # --- RECONCILED SUSPICIOUS LABELING (Block 4) ---
        
        # 1. Dynamic Risk (Parameterized Baseline & Sensitivity)
        r_sens    = opt_overrides.get(f"{scenario_type}_risk_sens_f", 0.4)
        r_offset  = opt_overrides.get(f"{scenario_type}_risk_offset", 1.0)
        r_neutral = opt_overrides.get(f"{scenario_type}_risk_neutral", 1.0)
        dynamic_risk = risk_score * (r_offset + (Total_Behavioral_Risk - r_neutral) * r_sens)


        # --- 2. ADAPTIVE ANOMALY THRESHOLD ---
        # In 2026, an 'anomaly' is relative. 
        # High sophistication actors (High Soph) have a HIGHER threshold for suspicion 
        # because their business is naturally complex.
        soph_buffer = 1.2 if actor['Sophistication'] == 'High' else 1.0
        effective_anomaly_threshold = 2.5 * soph_buffer * (region_mult)


        # 2. Optimized Thresholds
        # Replacing 1.2/1.0 with a generalized buffer; replacing region_mult with a weighted scalar
        s_buff_f = opt_overrides.get(f"{scenario_type}_soph_buffer_f", 1.2 if actor['Sophistication'] == 'High' else 1.0)
        r_weight = opt_overrides.get(f"{scenario_type}_region_weight_f", region_mult) # region_mult from Block 1
        a_base   = opt_overrides.get(f"{scenario_type}_anomaly_thresh_base", 2.5)
        
        effective_anomaly_threshold = a_base * s_buff_f * r_weight

        # --- 3. MULTI-PRONGED DETECTION TRIGGERS ---
        
        # Trigger A: Pure Behavioral Burst (Intensity-driven)
        is_high_intensity = (intensity > 2.8) 


        # 3. Decision Components
        # Replacing 2.8, 7.5, 2, and 3.0
        i_high_t  = opt_overrides.get(f"{scenario_type}_int_thresh_high", 2.8)
        
        is_high_intensity = (intensity > i_high_t)

        # Trigger B: Statistical Anomaly (Value-driven)
        # Does the monthly deviation exceed the risk-adjusted threshold?
        is_stat_anomaly = (monthly_deviation_factor > effective_anomaly_threshold and dynamic_risk > 7.5)

        risk_min  = opt_overrides.get(f"{scenario_type}_stat_risk_min", 7.5)
        is_stat_anomaly   = (monthly_deviation_factor > effective_anomaly_threshold and dynamic_risk > risk_min)


        # Trigger C: Obfuscation Red Flag (Network-driven)
        # Includes Shell Depth and Rail Hopping Intensity (new for 2026)
        is_structural_red_flag = (topo.get('shell_depth_offset', 0) > 2) or \
                                 (chan.get('rail_switch_intensity', 0) > 3.0)

        shell_t   = opt_overrides.get(f"{scenario_type}_shell_depth_thresh", 2)
        rail_t    = opt_overrides.get(f"{scenario_type}_rail_switch_thresh", 3.0)
        flag_floor = opt_overrides.get(f"{scenario_type}_red_flag_floor", 0)                                      
        shell_val = opt_overrides.get(f"{scenario_type}_shell_depth_offset", topo.get('shell_depth_offset', flag_floor))
        #rail_val  = chan.get('rail_switch_intensity', flag_floor)  rs_base = opt_overrides.get(f"{scenario_type}_rail_switch_base", chan['rail_switch_intensity'])
        is_structural_red_flag = (shell_val > shell_t) or (rs_base > rail_t)

        # --- 4. FINAL LABEL ASSIGNMENT ---
        #if is_high_intensity or is_stat_anomaly or is_structural_red_flag:
        # 4. Final Labeling
        # 'label_sensitivity' allows Optuna to force labels for specific edge cases
        l_sens = opt_overrides.get(f"{scenario_type}_label_sensitivity", 0)
        if is_high_intensity or is_stat_anomaly or is_structural_red_flag or (Total_Behavioral_Risk > l_sens > 0):
            
            current_label = 1 
        else:
            current_label = 0

        
        # -----------------------------------------------
        # 5. GENERATE TRANSACTIONS (TOPOLOGY & DIVERSITY)
        # -----------------------------------------------
        t_prev = anchor

        # --- 1. PRE-LOOP: Define the 2026 Bayesian Strategy ---
        # biz_vol_shift: Translates the mean upwards (1.1x to 1.5x) 
        # to simulate inflated "Shadow Revenue".
        biz_vol_shift = trial.suggest_float("biz_volume_mean_shift", 1.1, 1.5)

        # 1. Superseded direct trial calls (Maintaining names for efficiency)
        # biz_vol_shift: 1.1x to 1.5x (Shadow Revenue Inflation)
        biz_vol_shift = opt_overrides.get(f"{scenario_type}_biz_vol_shift", 1.3)


        # biz_vol_precision: Contracts the sigma (0.1 to 0.4) 
        # to simulate "Mechanical/Scripted" transaction consistency.
        biz_vol_precision = trial.suggest_float("biz_volume_sigma_scale", 0.1, 0.4)

        # biz_vol_precision: 0.1 to 0.4 (Mechanical Consistency)
        biz_vol_precision = opt_overrides.get(f"{scenario_type}_biz_vol_precision", 0.25)


        lower_bound = cfg['amt_min'] if 'amt_min' in cfg else 1.0
        upper_bound = cfg['amt_max'] if 'amt_max' in cfg else (cfg['amt_mu'] * 20)



        # 2. Optimized Amount Boundaries
        # Parameterizing hardcoded 1.0 and the 20x multiplier
        amt_min_p = opt_overrides.get(f"{scenario_type}_amt_min_base", cfg.get('amt_min', 1.0))
        amt_mu_p  = opt_overrides.get(f"{scenario_type}_amt_mu_base", cfg.get('amt_mu', 500.0))
        
        # Upper Bound Logic: Use 'amt_max' if present, else calculate with optimized multiplier
        mu_mult = opt_overrides.get(f"{scenario_type}_amt_mu_multiplier", 20.0)
        default_max = amt_mu_p * mu_mult
        
        lower_bound = amt_min_p
        upper_bound = opt_overrides.get(f"{scenario_type}_amt_max_base", cfg.get('amt_max', default_max))

        
        for _ in range(n_tx):
            # Amount calculation
            base_mu = cfg['amt_mu'] * actor_freq_multiplier * size_mult
            base_sigma = cfg['amt_sigma'] * soph_mult
            #amt = round(np.random.normal(base_mu, base_sigma) * intensity * monthly_deviation_factor, 2)


            # --- 2026 HYBRID AMOUNT LOGIC ---
            # We replace the original np.random.normal with the Bayesian-Shifted Truncated Normal
            
            # A. Translate Mean Upwards
            # Combined with the monthly_deviation_factor, this creates the 'Inflated' signature
            tuned_mu = base_mu * intensity * monthly_deviation_factor * biz_vol_shift

            
            # 1. Financial Magnitude (Amount)
            # Replaced isolated size/soph mults with the 2026 Risk Bridge
            m_base_f = opt_overrides.get(f"{scenario_type}_amt_mu_base_f", 1.0)
            s_base_f = opt_overrides.get(f"{scenario_type}_amt_sigma_base_f", 1.0)
            
            # Logic: Unified Risk * Intensity * Deviation * Shadow Shift
            # This ensures the amount reflects the ENTIRE suspicious profile
            tuned_mu = (amt_mu_p * Total_Behavioral_Risk * m_base_f) * intensity * monthly_deviation_factor * biz_vol_shift


            
            
            # B. Contract Standard Deviation (The Mechanical Signature)
            # We force a tight cluster around the inflated mean to show 'scripted' behavior
            tuned_sigma = base_sigma * biz_vol_precision
            safe_sigma = max(0.01, tuned_sigma)


            # Precision and Sigma Logic
            s_floor    = opt_overrides.get(f"{scenario_type}_sigma_floor", 0.01)
            base_sigma = (amt_mu_p * 0.1) * s_base_f # 10% baseline variance
            tuned_sigma = base_sigma * biz_vol_precision
            safe_sigma  = max(s_floor, tuned_sigma)

            
            # C. Calculate Truncation Bounds (Z-scores)
            # This ensures the deviation remains within system limits [min, max]
            a_vol = (lower_bound - tuned_mu) / safe_sigma
            b_vol = (upper_bound - tuned_mu) / safe_sigma

            # D. Final Generation (Replacing the original 'amt' line)
            # This produces a Low-Entropy, high-value signature for 2026 detection models
            amt = round(truncnorm.rvs(a_vol, b_vol, loc=tuned_mu, scale=safe_sigma), 2)
            
            # Final cashing out / retention adjustment
            # Economic Sensibility: Balance Retention Floor
            #amt = max(1.0, amt * (1 - econ['balance_retention_ratio']))

            # --- RECONCILED BALANCE RETENTION (Block 6) ---
            
            # 1. Fetch the Optimized Scalar (Defaults to 1.0)
            ret_f = opt_overrides.get(f"{scenario_type}_retention_scale_f", 1.0)
            
            # 2. Apply it to the Config Prior
            # Logic: amt * (1 - (Prior_Ratio * Optimized_Scalar))

            eff_retention = opt_overrides.get(
                f"{scenario_type}_retention_ratio_base", 
                econ['balance_retention_ratio']
            )            
            effective_retention = eff_retention * ret_f
            amt = max(1.0, amt * (1 - effective_retention))

            

            # 2. Retention Logic (Leakage)
            #amt = max(1.0, amt * (1 - econ.get('balance_retention_ratio', 0.01)))            
            
            # CHANNEL DIVERSITY: Rail Hopping and Latency Bypass
            is_rail_hop = np.random.random() < chan['hop_diversification_prob']
            v_bypass = chan['latency_bypass_multiplier'] if is_rail_hop else 1.0
            tx_type = np.random.choice(chan['rails']) if is_rail_hop else \
                      np.random.choice(['CREDIT', 'DEBIT'], p=multipliers['Occupation'].get(actor['Occupation'], [0.5, 0.5]) )

            # 3. Channel Diversity (Optimized Rail Hopping)
            h_prob_static = opt_overrides.get(f"{scenario_type}_hop_prob_static", 0.3)
            h_prob = h_prob_static * chan.get('hop_diversification_prob', 1.0)
            is_rail_hop = np.random.random() < h_prob

            if is_rail_hop:
                tx_type = np.random.choice(chan['rails'])
                
                #v_bypass = chan.get('latency_bypass_multiplier', 2.0)
                
                # 2. Optimized Latency Bypass (Replacing hardcoded 2.0)
                # This allows Optuna to find the exact speed boost for rail-hopping
                v_bypass = opt_overrides.get(f"{scenario_type}_v_bypass_base", chan.get('latency_bypass_multiplier', 2.0))
                
            else:
                # Simplified Fallback: Avoids index errors by using standard labels

                # 3. Streamlined Directional Labeling (2026 Holistic Approach)
                # Optuna tunes 'debit_prob_f' to find the best Credit/Debit balance for Recall
                d_prob = opt_overrides.get(f"{scenario_type}_debit_prob_f", 0.5)
                
                # Ensuring tx_type matches your 2026 legacy banking labels (CREDIT/DEBIT)
                tx_type = np.random.choice(['CREDIT', 'DEBIT'], p=[1.0 - d_prob, d_prob])
                
                # 2026 Upgrade: Optimized Neutral Bypass (replaces hardcoded 1.0)
                # Allows Optuna to tune the 'Standard' speed during a suspicious month
                v_bypass = opt_overrides.get(f"{scenario_type}_v_bypass_neutral", 1.0)
            
            
            # --- RESTORED & RETROFITTED TEMPORAL LOGIC ---
            # Baseline: Sophisticated actors move faster (original logic restored)
            base_gap = cfg['tx_gap_days'] / (1 + 0.2 * soph_mult)
            
            # 2026 Layer: Topology-driven compression (mule clusters/linkage accelerate flows)
            cluster_adj = 1.0 - (topo['inter-account_linkage'] * 0.5) if current_label == 1 else 1.0
            
            #gap_days = cfg['tx_gap_days'] * v_bypass * cluster_adj
            # Combined multidimensional gap calculation
            #gap_days = base_gap * v_bypass * cluster_adj

            #t_prev += timedelta(days=float(np.random.exponential(scale=gap_days)))

            # --- 2026 REFINEMENT: HOLISTIC VELOCITY ---
            # We ensure intensity also compresses the time gaps, 
            # as suspicious volume spikes rarely happen at 'Normal' business speeds.
            holistic_velocity_gain = (1 + 0.2 * soph_mult) * (1 / v_bypass) * intensity

            # 4. Spatiotemporal Velocity (Timing)
            # Replaces hardcoded 0.2 and 0.5 with optimized factors
            v_base    = opt_overrides.get(f"{scenario_type}_v_gain_base", 1.0)
            v_soph_f  = opt_overrides.get(f"{scenario_type}_v_gain_soph_f", 0.2)
        
            # Holistic Velocity Gain Logic
            # (1 + Factor * Sophistication) * (1 / Bypass) * Intensity
            #soph_val = 1.0 if actor['Sophistication'] == 'High' else 0.0
            # 1. Dynamically fetch the 'Relative Strength' from Config (No hardcoding in script)
            # This handles Low, Medium, High, or any new categories automatically
            soph_weight = get_factor(multipliers.get('Sophistication_Weights'), actor['Sophistication'])
            
            velocity_gain = (v_base + v_soph_f * soph_weight) * (1 / v_bypass) * intensity

            # Cluster Adjustment: How much network linkage compresses the time between tx
            c_link_f  = opt_overrides.get(f"{scenario_type}_cluster_link_f", 0.5)            
            cluster_adj = 1.0 - (topo.get('inter-account_linkage', 0) * c_link_f) if current_label == 1 else 1.0            
            
            gap_days = cfg['tx_gap_days'] / max(0.1, holistic_velocity_gain * cluster_adj)

            # Final Temporal Spacing (Exponential Gap)
            g_base    = opt_overrides.get(f"{scenario_type}_gap_days_base", cfg.get('tx_gap_days', 1.0))
            g_floor   = opt_overrides.get(f"{scenario_type}_gap_min_floor", 0.1)
            
            gap_days = g_base / max(g_floor, velocity_gain * cluster_adj)
            
            t_prev += timedelta(days=float(np.random.exponential(scale=gap_days)))
            
            #rows.append(_assemble_tx(acct, party_key, t_prev, amt, tx_type, config))
            rows.append(_assemble_tx(acct, party_key, ref_date, t_prev, amt, tx_type, cfg, label=current_label))
            
        # -----------------------------------------------
        # 6. MICRO-TRANSACTIONS (Final Fan-out)
        # -----------------------------------------------
        # Scaled by 2026 micro_multiplier and structural intensity
        #if current_label == 1:
            #rows.extend(_generate_micro_tx(
                #t_prev, amt, config, ref_date=ref_date, label=current_label,
                #multiplier_override=int(cfg['micro_multiplier'] * intensity)
            #))
            #rows.extend(_generate_micro_tx(t_prev, amt, cfg, ref_date=ref_date, label=current_label))

        # --- REFINED POST-PROCESSING (2026 NOISE COMPLIANT) ---
        
        # 1. Base probability for micro-transactions (legitimate business 'dust')
        #micro_tx_prob = 0.15 
        
        # 2. Economic & Topology Boost: Suspicious actors or those in 
        # complex networks (mule clusters) have higher 'layering' dust
        #if current_label == 1 or topo['mule_cluster_density'] > 0.5:
            #micro_tx_prob += 0.35 

        # --- RECONCILED MICRO-TRANSACTION LOGIC (Block 8) ---
        
        # 1. Resolve Parameters
        # Baseline: 0.15; Boost: 0.35; Cluster Threshold: 0.5
        #m_base   = opt_overrides.get(f"{scenario_type}_micro_tx_base", 0.15)
        #m_boost  = opt_overrides.get(f"{scenario_type}_micro_tx_boost", 0.35)
        #d_thresh = opt_overrides.get(f"{scenario_type}_mule_density_threshold", 0.5)

        # 2. Optimized Network Input (Replacing the hardcoded topo.get default)
        # Optuna can now override the raw 'mule_cluster_density' from the network topology
        #m_density = opt_overrides.get(f"{scenario_type}_mule_density_base", topo.get('mule_cluster_density', 0))

        # 2. Optimized Probability Calculation
        # Logic: Base probability + Boost if actor is suspicious OR in a high-density cluster
        #micro_tx_prob = m_base
        
        #if current_label == 1 or m_density > d_thresh:
            #micro_tx_prob += m_boost

            
        # 3. Generate micro-transactions based on probability, not just the label
        # This forces the ML model to learn the difference between 'legitimate dust' and 'illicit layering'
        #if np.random.random() < micro_tx_prob:
            # We still pass current_label so the individual micro-tx rows are tagged correctly for training
            #rows.extend(_generate_micro_tx(t_prev, amt, cfg, ref_date=ref_date, label=current_label))

        

