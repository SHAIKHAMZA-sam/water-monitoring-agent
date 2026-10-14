"""Create a SYNTHETIC water-quality dataset for training the model.

Why synthetic? Real labelled data needs lab testing. This file creates
realistic-looking readings so the project runs anywhere with no download.
For a real deployment, replace data/water_quality.csv with real lab data that
has the same columns: ph, turbidity, tds, chlorine, potable (1 = safe, 0 = unsafe).
"""
import numpy as np
import pandas as pd

from config import DATA_FILE, LIMITS


def generate(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    good = rng.random(n) < 0.55  # 55% well-managed supplies, 45% problem supplies

    ph = np.where(good, rng.normal(7.3, 0.35, n), rng.uniform(5.0, 10.0, n))
    turbidity = np.where(good, rng.gamma(2.0, 0.2, n), rng.gamma(2.5, 1.2, n))
    tds = np.where(good, rng.normal(300, 80, n), rng.normal(550, 250, n))
    chlorine = np.where(good, rng.normal(0.6, 0.3, n), rng.gamma(1.5, 0.4, n))

    ph = np.clip(ph, 3, 12)
    turbidity = np.clip(turbidity, 0, None)
    tds = np.clip(tds, 20, None)
    chlorine = np.clip(chlorine, 0, None)

    # Label from the TRUE values: potable only if every reading is in range.
    potable = np.ones(n, dtype=int)
    for name, values in [("ph", ph), ("turbidity", turbidity), ("tds", tds), ("chlorine", chlorine)]:
        low, high = LIMITS[name]
        potable &= ((values >= low) & (values <= high)).astype(int)

    # Sensors are not perfect: add measurement noise AFTER labelling.
    ph = ph + rng.normal(0, 0.10, n)
    turbidity = np.clip(turbidity + rng.normal(0, 0.10, n), 0, None)
    tds = np.clip(tds + rng.normal(0, 15, n), 0, None)
    chlorine = np.clip(chlorine + rng.normal(0, 0.05, n), 0, None)

    # A few wrong labels, like real lab records.
    flip = rng.random(n) < 0.03
    potable = np.where(flip, 1 - potable, potable)

    return pd.DataFrame({
        "ph": ph.round(2), "turbidity": turbidity.round(2),
        "tds": tds.round(1), "chlorine": chlorine.round(2), "potable": potable,
    })


if __name__ == "__main__":
    df = generate()
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} rows to {DATA_FILE}")
    print("Safe (1) vs unsafe (0) counts:")
    print(df["potable"].value_counts().to_string())
