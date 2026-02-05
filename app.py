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
