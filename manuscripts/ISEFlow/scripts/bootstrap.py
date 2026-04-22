import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def kolmogorov_smirnov(x1, x2, alternative="two-sided", method="auto"):
    """
    Two-sample Kolmogorov-Smirnov test.

    Args:
        x1, x2: Arrays of sample values.
        alternative (str): 'two-sided', 'less', or 'greater'.
        method (str): 'auto', 'exact', or 'asymp'.

    Returns:
        tuple: (KS statistic, p-value)
    """
    res = ks_2samp(x1, x2, alternative=alternative, method=method)
    return res.statistic, res.pvalue


def load_rcp_samples(path):
    df = pd.read_csv(path).copy()

    # Normalize scenario column name if needed
    if "Scenario" in df.columns and "scenarios" not in df.columns:
        df["scenarios"] = df["Scenario"]
    
    if 'sle' in df.columns:
        df['true'] = df['sle']

    if "scenarios" not in df.columns:
        raise ValueError(f"'scenarios' column not found in {path}")

    if "pred" not in df.columns:
        raise ValueError(f"'pred' column not found in {path}")

    # Normalize scenario labels just in case
    df["scenarios"] = df["scenarios"].astype(str).str.strip().str.lower()

    rcp85 = df.loc[df["scenarios"] == "rcp8.5", "true"].to_numpy()
    rcp26 = df.loc[df["scenarios"] == "rcp2.6", "true"].to_numpy()

    if len(rcp85) == 0 or len(rcp26) == 0:
        raise ValueError(
            f"Missing one or both scenarios in {path}: "
            f"len(rcp8.5)={len(rcp85)}, len(rcp2.6)={len(rcp26)}"
        )

    return rcp85, rcp26


def bootstrap_ks(path, label, n_bootstrap=10000, seed=0):
    """
    Compute observed two-sample KS statistic, p-value, and bootstrap CI
    by resampling each scenario-specific prediction distribution.

    Returns:
        dict with keys:
            label, d_obs, p_obs, ci_lower, ci_upper, bootstrap_ds
    """
    rcp85, rcp26 = load_rcp_samples(path)

    # Observed KS statistic and p-value
    d_obs, p_obs = kolmogorov_smirnov(rcp85, rcp26)

    rng = np.random.default_rng(seed)
    bootstrap_ds = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        boot85 = rng.choice(rcp85, size=len(rcp85), replace=True)
        boot26 = rng.choice(rcp26, size=len(rcp26), replace=True)
        d_boot, _ = kolmogorov_smirnov(boot85, boot26)
        bootstrap_ds[i] = d_boot

    ci_lower, ci_upper = np.quantile(bootstrap_ds, [0.025, 0.975])

    result = {
        "label": label,
        "d_obs": d_obs,
        "p_obs": p_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_ds": bootstrap_ds,
    }

    print(
        f"{label}: KS D = {d_obs:.3f}, p = {p_obs:.3e} "
        f"(95% bootstrap CI: {ci_lower:.3f}–{ci_upper:.3f})"
    )

    return result


def bootstrap_ks_difference(path_a, label_a, path_b, label_b, n_bootstrap=10000, seed=0):
    """
    Bootstrap the difference in KS D between two emulators / models.

    This directly addresses whether the difference in D between models
    is distinguishable from zero under resampling.

    Returns:
        dict with keys:
            label_a, label_b,
            d_a, d_b, delta_obs,
            ci_lower, ci_upper,
            p_bootstrap_two_sided,
            delta_bootstrap
    """
    a85, a26 = load_rcp_samples(path_a)
    b85, b26 = load_rcp_samples(path_b)

    d_a, _ = kolmogorov_smirnov(a85, a26)
    d_b, _ = kolmogorov_smirnov(b85, b26)
    delta_obs = d_a - d_b

    rng = np.random.default_rng(seed)
    delta_bootstrap = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        boot_a85 = rng.choice(a85, size=len(a85), replace=True)
        boot_a26 = rng.choice(a26, size=len(a26), replace=True)
        boot_b85 = rng.choice(b85, size=len(b85), replace=True)
        boot_b26 = rng.choice(b26, size=len(b26), replace=True)

        d_a_boot, _ = kolmogorov_smirnov(boot_a85, boot_a26)
        d_b_boot, _ = kolmogorov_smirnov(boot_b85, boot_b26)

        delta_bootstrap[i] = d_a_boot - d_b_boot

    ci_lower, ci_upper = np.quantile(delta_bootstrap, [0.025, 0.975])

    # Bootstrap two-sided p-value for H0: delta = 0
    # Approximate using the proportion of bootstrap draws on the opposite side of zero.
    if delta_obs >= 0:
        tail_prob = np.mean(delta_bootstrap <= 0)
    else:
        tail_prob = np.mean(delta_bootstrap >= 0)
    p_bootstrap_two_sided = min(1.0, 2 * tail_prob)

    result = {
        "label_a": label_a,
        "label_b": label_b,
        "d_a": d_a,
        "d_b": d_b,
        "delta_obs": delta_obs,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_bootstrap_two_sided": p_bootstrap_two_sided,
        "delta_bootstrap": delta_bootstrap,
    }

    print(
        f"{label_a} - {label_b}: ΔKS D = {delta_obs:.3f} "
        f"(95% bootstrap CI: {ci_lower:.3f}–{ci_upper:.3f}), "
        f"bootstrap p = {p_bootstrap_two_sided:.3e}"
    )

    if ci_lower > 0 or ci_upper < 0:
        print("  -> Difference excludes 0: statistically distinguishable under bootstrap resampling.")
    else:
        print("  -> Difference includes 0: not clearly distinguishable under bootstrap resampling.")

    return result


# -------------------------------------------------------------------
# Individual KS results
# -------------------------------------------------------------------

ais_iseflow = bootstrap_ks(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/iseflow-ais-predictions.csv",
    "ISEFlow AIS",
    n_bootstrap=1000,
    seed=0,
)

gris_iseflow = bootstrap_ks(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/iseflow-gris-predictions.csv",
    "ISEFlow GrIS",
    n_bootstrap=1000,
    seed=1,
)

exit()

ais_gp = bootstrap_ks(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/emulandice-ais-predictions.csv",
    "GP emulator AIS",
    n_bootstrap=1000,
    seed=2,
)

gris_gp = bootstrap_ks(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/emulandice-gris-predictions.csv",
    "GP emulator GrIS",
    n_bootstrap=1000,
    seed=3,
)

# -------------------------------------------------------------------
# Difference-in-KS results: this is the key reviewer-facing test
# -------------------------------------------------------------------

ais_diff = bootstrap_ks_difference(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/iseflow-ais-predictions.csv",
    "ISEFlow AIS",
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/emulandice-ais-predictions.csv",
    "GP emulator AIS",
    n_bootstrap=1000,
    seed=10,
)

gris_diff = bootstrap_ks_difference(
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/iseflow-gris-predictions.csv",
    "ISEFlow GrIS",
    r"/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow/emulandice-gris-predictions.csv",
    "GP emulator GrIS",
    n_bootstrap=1000,
    seed=11,
)