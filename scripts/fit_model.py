"""Fit a weighted multinomial model of BMI category on BRFSS 2024."""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

SRC = "/mnt/user-data/uploads/brfss2024_slim.csv"

# --- Value labels. Missing/refused codes map to an explicit "Not given"
# --- level so a site visitor can skip a question without being dropped.
LEVELS = {
    "sex": {1: "Male", 2: "Female"},
    "race": {1: "White", 2: "Black", 3: "Asian", 4: "Native American",
             5: "Hispanic", 6: "Other"},
    "education": {1: "Did not finish high school", 2: "High school graduate",
                  3: "Some college", 4: "College graduate"},
    "income": {1: "Under $15k", 2: "$15k-$25k", 3: "$25k-$35k", 4: "$35k-$50k",
               5: "$50k-$100k", 6: "$100k-$200k", 7: "$200k+"},
    "marital": {1: "Married", 2: "Divorced", 3: "Widowed", 4: "Separated",
                5: "Never married", 6: "Unmarried couple"},
    "children": {1: "No children", 2: "One child", 3: "Two children",
                 4: "Three children", 5: "Four children", 6: "Five or more"},
    "employment": {1: "Employed for wages", 2: "Self-employed",
                   3: "Out of work 1+ years", 4: "Out of work under 1 year",
                   5: "Homemaker", 6: "Student", 7: "Retired",
                   8: "Unable to work"},
    "urbanicity": {1: "Urban", 2: "Rural"},
    "metro": {1: "Metro county", 2: "Non-metro county"},
    "housing": {1: "Own", 2: "Rent", 3: "Other arrangement"},
    "veteran": {1: "Veteran", 2: "Not a veteran"},
    "activity": {1: "Exercised in past 30 days", 2: "No exercise in past 30 days"},
    "smoking": {1: "Smokes daily", 2: "Smokes some days", 3: "Former smoker",
                4: "Never smoked"},
    "binge": {1: "No binge drinking", 2: "Binge drinks"},
    "language": {1: "English", 2: "Spanish"},
}

# BRFSS sentinel codes that mean don't-know / refused, by source column.
BAD = {
    "_EDUCAG": [9], "_INCOMG1": [9], "MARITAL": [9], "_CHLDCNT": [9],
    "EMPLOY1": [9], "RENTHOM1": [7, 9], "VETERAN3": [7, 9],
    "_TOTINDA": [9], "_SMOKER3": [9], "_RFBING6": [9], "QSTLANG": [3],
}

SRC_COL = {
    "sex": "SEXVAR", "race": "_IMPRACE", "education": "_EDUCAG",
    "income": "_INCOMG1", "marital": "MARITAL", "children": "_CHLDCNT",
    "employment": "EMPLOY1", "urbanicity": "_URBSTAT", "metro": "_METSTAT",
    "housing": "RENTHOM1", "veteran": "VETERAN3", "activity": "_TOTINDA",
    "smoking": "_SMOKER3", "binge": "_RFBING6", "language": "QSTLANG",
}

AGE_BANDS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
    12: "75-79", 13: "80+",
}

OUTCOMES = ["Underweight", "Normal weight", "Overweight", "Obese"]

FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY", 66: "Guam", 72: "Puerto Rico", 78: "US Virgin Islands",
}


def build():
    df = pd.read_csv(SRC, low_memory=False)
    print(f"loaded {len(df):,}")

    out = pd.DataFrame(index=df.index)

    # Outcome: 1=under 2=normal 3=overweight 4=obese
    out["y"] = df["_BMI5CAT"].astype(int) - 1

    # Survey weight, normalised so regularisation behaves predictably.
    out["w"] = df["_LLCPWT"] / df["_LLCPWT"].mean()

    # Categorical predictors -> string labels, sentinels -> "Not given"
    for name, col in SRC_COL.items():
        s = df[col].copy()
        for code in BAD.get(col, []):
            s = s.replace(code, np.nan)
        out[name] = s.map(LEVELS[name]).fillna("Not given")

    out["age"] = df["_AGEG5YR"].replace(14, np.nan).map(AGE_BANDS).fillna("Not given")
    out["state"] = df["_STATE"].map(FIPS).fillna("Other")

    # Drinks per week: 2 implied decimals, 99900 is the missing code.
    dw = df["_DRNKWK3"].replace(99900, np.nan) / 100.0
    out["drinks"] = pd.cut(
        dw, [-0.01, 0.01, 1, 3, 7, 14, 1e6],
        labels=["None", "Under 1/week", "1-3/week", "3-7/week",
                "7-14/week", "14+/week"],
    ).astype(object)
    out["drinks"] = out["drinks"].fillna("Not given")

    return out


CATS = (list(SRC_COL) + ["age", "state", "drinks"])

# Reference level per variable: the modal / natural baseline.
REF = {
    "sex": "Male", "race": "White", "education": "College graduate",
    "income": "$50k-$100k", "marital": "Married", "children": "No children",
    "employment": "Employed for wages", "urbanicity": "Urban",
    "metro": "Metro county", "housing": "Own", "veteran": "Not a veteran",
    "activity": "Exercised in past 30 days", "smoking": "Never smoked",
    "binge": "No binge drinking", "language": "English",
    "age": "40-44", "state": "OH", "drinks": "None",
}


def design(d):
    """One-hot with an explicit reference level dropped per variable."""
    blocks, names = [], []
    for c in CATS:
        levels = sorted(x for x in d[c].unique() if x != REF[c])
        for lv in levels:
            blocks.append((d[c] == lv).to_numpy(np.float32))
            names.append(f"{c}={lv}")
    # Sex interactions -- the Economist model leaned on these.
    sexf = (d["sex"] == "Female").to_numpy(np.float32)
    for c in ["race", "age", "marital", "children"]:
        for lv in sorted(x for x in d[c].unique() if x != REF[c]):
            blocks.append(sexf * (d[c] == lv).to_numpy(np.float32))
            names.append(f"sex=Female*{c}={lv}")
    return np.column_stack(blocks), names


if __name__ == "__main__":
    d = build()
    X, names = design(d)
    y = d["y"].to_numpy()
    w = d["w"].to_numpy()
    print(f"design: {X.shape[0]:,} x {X.shape[1]}")

    rng = np.random.default_rng(0)
    test = rng.random(len(y)) < 0.2
    tr = ~test

    model = LogisticRegression(C=1e4, max_iter=3000, n_jobs=-1)
    model.fit(X[tr], y[tr], sample_weight=w[tr])

    p = model.predict_proba(X[test])
    base = np.average(
        pd.get_dummies(y[tr]).to_numpy(), axis=0, weights=w[tr]
    )
    ll = log_loss(y[test], p, sample_weight=w[test], labels=[0, 1, 2, 3])
    ll0 = log_loss(y[test], np.tile(base, (test.sum(), 1)),
                   sample_weight=w[test], labels=[0, 1, 2, 3])
    print(f"\nweighted log loss: {ll:.4f}   baseline: {ll0:.4f}"
          f"   improvement: {100*(1-ll/ll0):.1f}%")

    pred = np.average(p, axis=0, weights=w[test])
    actual = np.average(pd.get_dummies(y[test]).to_numpy(), axis=0,
                        weights=w[test])
    print("\ncalibration (weighted, held-out):")
    for i, o in enumerate(OUTCOMES):
        print(f"  {o:<15} predicted {100*pred[i]:5.1f}%   actual {100*actual[i]:5.1f}%")

    full = LogisticRegression(C=1e4, max_iter=3000, n_jobs=-1)
    full.fit(X, y, sample_weight=w)

    coefs = {n: [round(float(full.coef_[k][j]), 5) for k in range(4)]
             for j, n in enumerate(names)}
    bundle = {
        "outcomes": OUTCOMES,
        "intercept": [round(float(v), 5) for v in full.intercept_],
        "reference": REF,
        "levels": {c: sorted(d[c].unique().tolist()) for c in CATS},
        "coefficients": coefs,
        "n": int(len(y)),
        "source": "CDC BRFSS 2024, weighted by _LLCPWT",
    }
    with open("/home/claude/model.json", "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    print(f"\nwrote model.json ({len(names)} coefficients)")
