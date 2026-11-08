"""
Train the SaltBae salt-dose model.

Output: saltbae_rf_model.joblib  (same file name and same 3 input columns as
before, so python/main.py works unchanged).

Two ways to run it:

  python train_salt_model.py
      Trains on synthetic data made from the formula below (your original
      formula, but with a wider ice range and NO cap at 35 g).

  python train_salt_model.py --data my_measurements.csv
      Trains on REAL measurements. The CSV needs these columns:
      ice_thickness_mm, pavement_temp_c, humidity_pct, required_salt_grams

IMPORTANT: a model trained on synthetic data can only be as good as the
formula that made the data. Its "accuracy" is how closely it copies that
formula. To be accurate about the real world, train on real measurements.
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURES = ["ice_thickness_mm", "pavement_temp_c", "humidity_pct"]
TARGET = "required_salt_grams"
OUTPUT_PATH = "saltbae_rf_model.joblib"

# Ranges the model is trained on. Outside these, tree models go flat.
ICE_MAX_MM = 40.0      # wider than before (was 15)
TEMP_RANGE = (-15.0, 2.0)
HUM_RANGE = (40.0, 100.0)
N_SAMPLES = 20000
SEED = 42


def needed_salt_grams(ice, temp, hum):
    """Your original formula, noise-free, and without the 35 g cap."""
    ice = np.asarray(ice, dtype=float)
    temp = np.asarray(temp, dtype=float)
    hum = np.asarray(hum, dtype=float)
    temp_factor = np.where(temp < 0, np.abs(temp) * 0.45, 0.0)
    grams = ice * 1.4 + ice * temp_factor + (hum / 100.0) * 1.2
    grams = np.where(ice < 1.0, 0.0, grams)    # too little ice: no salt
    grams = np.where(temp > 1.0, 0.0, grams)   # too warm: no salt
    return np.clip(grams, 0.0, None)


def make_synthetic_data(n=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    # Most real ice is thin, so spend more samples there:
    # 60% of samples 0-15 mm, 40% of samples 15-ICE_MAX_MM
    n_thin = int(n * 0.6)
    ice = np.concatenate([
        rng.uniform(0.0, 15.0, n_thin),
        rng.uniform(15.0, ICE_MAX_MM, n - n_thin),
    ])
    temp = rng.uniform(*TEMP_RANGE, n)
    hum = rng.uniform(*HUM_RANGE, n)
    grams = needed_salt_grams(ice, temp, hum) + rng.normal(0, 0.3, n)
    grams = np.clip(grams, 0.0, None)  # no negative doses, but NO upper cap
    return pd.DataFrame({
        "ice_thickness_mm": ice,
        "pavement_temp_c": temp,
        "humidity_pct": hum,
        TARGET: grams,
    })


def candidate_models():
    # Kept small on purpose: the model file has to load on the UNO Q.
    return {
        "RandomForest (small)": RandomForestRegressor(
            n_estimators=60, max_depth=14, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=10, random_state=SEED),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV of real measurements (optional)")
    args = parser.parse_args()

    real_data = args.data is not None
    if real_data:
        df = pd.read_csv(args.data)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            raise SystemExit(f"CSV is missing columns: {missing}")
        df = df[FEATURES + [TARGET]].dropna()
        print(f"Loaded {len(df)} real measurements from {args.data}")
        if len(df) < 200:
            print("WARNING: fewer than 200 rows. Expect a noisy model.")
    else:
        df = make_synthetic_data()
        print(f"Generated {len(df)} synthetic samples "
              f"(ice 0-{ICE_MAX_MM:.0f} mm, no 35 g cap)")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # Pick the best model type using 5-fold cross-validation on the training set
    print("\nComparing models (5-fold cross-validated RMSE, lower is better):")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for name, model in candidate_models().items():
        rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="neg_root_mean_squared_error").mean()
        scores[name] = rmse
        print(f"  {name:22s} RMSE = {rmse:.3f} g")
    best_name = min(scores, key=scores.get)
    print(f"-> Using {best_name}")

    model = candidate_models()[best_name]
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    print("\nHeld-out test set:")
    print(f"  R2   = {r2_score(y_test, pred):.4f}")
    print(f"  RMSE = {rmse:.3f} g")
    print(f"  MAE  = {mean_absolute_error(y_test, pred):.3f} g")

    if not real_data:
        # How close is the model to the exact formula (not the noisy labels)?
        # Random points are used on purpose: the formula jumps to 0 below 1 mm
        # of ice and above 1 C, and no model can match a jump exactly, so
        # points right next to those cutoffs always show a big error.
        rng = np.random.default_rng(123)

        def check(label, ice_max, temp_min):
            n = 20000
            g = pd.DataFrame({
                "ice_thickness_mm": rng.uniform(0, ice_max, n),
                "pavement_temp_c": rng.uniform(temp_min, TEMP_RANGE[1], n),
                "humidity_pct": rng.uniform(*HUM_RANGE, n),
            })
            truth = needed_salt_grams(g.ice_thickness_mm, g.pavement_temp_c,
                                      g.humidity_pct)
            err = np.abs(model.predict(g) - truth)
            print(f"  {label}: mean {err.mean():.2f} g | "
                  f"95% of points within {np.percentile(err, 95):.2f} g | "
                  f"99% within {np.percentile(err, 99):.2f} g")

        print("\nError vs the exact formula (random points):")
        check("everyday  (ice<=15 mm, temp>=-10 C)", 15.0, -10.0)
        check("full range (ice<=40 mm, temp>=-15 C)", ICE_MAX_MM, TEMP_RANGE[0])

    print("\nSanity checks (-3 C, 80% humidity):")
    for ice in (1.5, 3, 5, 8, 12, 20, 30, 40):
        x = pd.DataFrame([[ice, -3.0, 80.0]], columns=FEATURES)
        line = f"  {ice:5.1f} mm ice -> {model.predict(x)[0]:7.2f} g"
        if not real_data:
            line += f"   (formula: {float(needed_salt_grams(ice, -3.0, 80.0)):7.2f} g)"
        print(line)

    joblib.dump(model, OUTPUT_PATH)
    print(f"\nSaved model to: {OUTPUT_PATH}")
    import sklearn
    print(f"Trained with scikit-learn {sklearn.__version__} - pin the same "
          f"version in python/requirements.txt")


if __name__ == "__main__":
    main()
