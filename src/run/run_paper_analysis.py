#!/usr/bin/env python3
"""
Master analysis script for the paper:
"The Value Atlas of AI: Mapping World Human Values in Large Language Models"

Covers Study 1-4 as defined in advisor_paper_original.md.
All statistical tests are computed and reported uniformly.

Usage:
    python src/run/run_paper_analysis.py                  # run everything (from precomputed PCA)
    python src/run/run_paper_analysis.py --study 1        # run Study 1 only
    python src/run/run_paper_analysis.py --study 2 3      # run Study 2 & 3
    python src/run/run_paper_analysis.py --model-imitation  # model imitation only
    python src/run/run_paper_analysis.py --regression-only # only generate regression CSV
    python src/run/run_paper_analysis.py --rebuild-from-raw # rebuild PCA from raw interviews, then run all

Outputs:
    results/analysis/study1_intrinsic_bias.json
    results/analysis/study2_english_advantage.json
    results/analysis/study3_digital_orientalism.json
    results/analysis/study4_colonial_legacies.json
    results/analysis/regression_data_v5.csv
    results/paper_data/regression_data.csv
    results/analysis/paper_statistics_all.json
"""

import sys
import json
import argparse
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DATA_DIR = PROJECT_ROOT / "results" / "paper_data"
PAPER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model name normalisation (API format -> folder format)
# ---------------------------------------------------------------------------
MODEL_NAME_MAP = {
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "deepseek/deepseek-chat-v3.1": "deepseek-chat-v3.1",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-3-pro-preview": "gemini-3-pro-preview",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "meta-llama/llama-3.2-3b-instruct": "llama-3.2-3b-instruct",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "microsoft/phi-3-mini-128k-instruct": "phi-3-mini-128k-instruct",
    "mistralai/mistral-medium-3.1": "mistral-medium-3.1",
    "mistralai/mistral-nemo": "mistral-nemo",
    "openai/gpt-5.1": "gpt-5.1",
    "qwen/qwen3-max": "qwen3-max",
    "x-ai/grok-4.1-fast": "grok-4.1-fast",
}

# The released paper dataset already contains the final 20-model roleplay sample.
EXCLUDED_ROLEPLAY = set()

# English-native countries (excluded from English-advantage calculation)
# Hong Kong excluded: native language is zh-hk (Cantonese), not English
# This gives 20 EN-native + 46 non-English = 66 total
EN_NATIVE_COUNTRIES = {
    "Australia", "Canada", "Ghana", "Ireland", "Kenya",
    "Malaysia", "Malta", "New Zealand", "Nigeria", "Pakistan",
    "Philippines", "Puerto Rico", "Rwanda", "Singapore",
    "South Africa", "Trinidad and Tobago", "United Kingdom",
    "United States of America", "Zambia", "Zimbabwe",
}

# Model origin mapping
MODEL_ORIGIN = {
    "claude-3-7-sonnet-20250219": "US", "claude-sonnet-4.5": "US",
    "gpt-4o": "US", "gpt-4o-mini": "US", "gpt-5.1": "US",
    "gemini-2.5-flash": "US", "gemini-2.5-pro": "US",
    "gemini-3-pro-preview": "US", "gemma-3-4b-it": "US",
    "llama-3.2-3b-instruct": "US", "llama-3.3-70b-instruct": "US",
    "grok-4.1-fast": "US", "phi-3-mini-128k-instruct": "US",
    "deepseek-chat": "China", "deepseek-chat-v3.1": "China",
    "doubao-1-5-pro-32k-250115": "China", "kimi-k2": "China",
    "qwen3-max": "China",
    "mistral-medium-3.1": "Europe", "mistral-nemo": "Europe",
}

# Inglehart-Welzel cultural region classification for the 66 WVS countries
CULTURAL_REGION_OVERRIDE = {
    "African-Islamic": "African-Islamic",
    "Catholic Europe": "Catholic Europe",
    "Confucian": "Confucian",
    "Latin America": "Latin America",
    "Orthodox Europe": "Orthodox Europe",
    "Protestant Europe": "Protestant Europe",
    "English-Speaking": "English-Speaking",
    "West & South Asia": "West & South Asia",
}

# 12 Arabic countries for French advantage
ARABIC_COUNTRIES_12 = [
    "Algeria", "Egypt", "Iraq", "Jordan", "Kuwait", "Lebanon",
    "Libya", "Morocco", "Palestine", "Qatar", "Tunisia", "Yemen",
]

# Language family mapping
LANGUAGE_FAMILY = {
    "ar": "Semitic", "zh-cn": "CJK", "zh-tw": "CJK", "zh-hk": "CJK",
    "ja": "CJK", "ko": "CJK", "ru": "Slavic", "es": "Romance",
    "pt": "Romance", "fr": "Romance", "it": "Romance", "de": "Germanic",
    "en": "Germanic", "en-native": "Germanic",
}

# Confucian entities for Study 4 (use exact IVS names)
CONFUCIAN_ENTITIES = ["China", "Japan", "Korea, Republic of", "Taiwan, Province of China", "Hong Kong", "Macao"]

# Sub-Saharan Africa British colonies
AFRICA_BRITISH = ["Kenya", "Nigeria", "Ghana", "South Africa", "Zimbabwe", "Zambia"]
# Sub-Saharan Africa French colonies
AFRICA_FRENCH = ["Mali", "Burkina Faso"]

# Latin America groupings
LATAM_SPANISH = [
    "Argentina", "Bolivia", "Chile", "Colombia", "Ecuador", "Guatemala",
    "Mexico", "Nicaragua", "Peru", "Uruguay", "Venezuela",
]
LATAM_PORTUGUESE = ["Brazil"]
LATAM_FRENCH = ["Haiti"]


# ===========================================================================
# Data loading helpers
# ===========================================================================

def load_ivs_coordinates() -> dict:
    path = PROJECT_ROOT / "data" / "country_values" / "country_scores_pca.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    coords = {}
    for item in data:
        c = item.get("Country")
        pc1, pc2 = item.get("PC1_rescaled"), item.get("PC2_rescaled")
        if c and pc1 is not None and pc2 is not None:
            coords[c] = {
                "PC1": float(pc1),
                "PC2": float(pc2),
                "cultural_region": item.get("Cultural Region", ""),
                "is_islamic": item.get("Islamic", False),
            }
    return coords


def load_intrinsic_pca() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "llm_pca" / "intrinsic" / "llm_pca_entity_scores.pkl"
    df = pd.read_pickle(path)
    df["model_name"] = df["model_name"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    return df


def load_roleplay_pca() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "llm_pca" / "multilingual" / "roleplay_ml_pca_entity_scores_latest.pkl"
    df = pd.read_pickle(path)
    df["model_name"] = df["model_name"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    return df


def load_pca_model() -> dict:
    path = PROJECT_ROOT / "data" / "country_values" / "pca_model_fixed.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def euclidean(a_pc1, a_pc2, b_pc1, b_pc2) -> float:
    return float(np.sqrt((a_pc1 - b_pc1) ** 2 + (a_pc2 - b_pc2) ** 2))


def match_country(country: str, ivs_coords: dict):
    if not isinstance(country, str):
        return None
    if country in ivs_coords:
        return country
    cl = country.lower().strip()
    for name in ivs_coords:
        if name.lower().strip() == cl:
            return name
    return None


def country_ea_paper_method(subset: pd.DataFrame) -> pd.Series:
    """Paper's aggregation: average distances per country, then compute EA."""
    agg = subset.groupby("country").agg(
        mean_d_native=("d_native", "mean"), mean_d_english=("d_english", "mean"),
    ).reset_index()
    agg["ea"] = (agg["mean_d_native"] - agg["mean_d_english"]) / agg["mean_d_native"] * 100
    return agg.set_index("country")["ea"]


def entity_ea_paper(subset: pd.DataFrame) -> float:
    """Paper method for a single entity: average distances, then compute EA."""
    if subset.empty:
        return np.nan
    return float(
        (subset["d_native"].mean() - subset["d_english"].mean())
        / subset["d_native"].mean() * 100
    )


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cohens_d_paired(diff):
    m = np.mean(diff)
    s = np.std(diff, ddof=1)
    if s == 0:
        return 0.0
    return float(m / s)


def fmt(v, decimals=3):
    if isinstance(v, float):
        return round(v, decimals)
    return v


# ===========================================================================
# Study 1: Pervasive Secular-Rational Bias (Intrinsic)
# ===========================================================================

def run_study1(intrinsic_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("STUDY 1: Pervasive Secular-Rational Bias")
    print("=" * 70)

    llm = intrinsic_df[intrinsic_df["data_source"] != "IVS"].copy()
    llm = llm[~llm["model_name"].isin(EXCLUDED_ROLEPLAY)]

    # Intrinsic PCA encodes language as the last segment of country_code,
    # e.g. "LLM_llm_anthropic_claude-sonnet-4.5_ar" -> "ar"
    if "language" not in llm.columns:
        code_col = "country_code_original" if "country_code_original" in llm.columns else "country_code"
        llm["language"] = llm[code_col].astype(str).str.rsplit("_", n=1).str[-1]

    n_models = llm["model_name"].nunique()
    models = sorted(llm["model_name"].unique())
    n_combinations = len(llm)

    print(f"  Models: {n_models}")
    print(f"  Model-language combinations: {n_combinations}")

    # PC1 = Secular-Rational (Traditional vs Secular-Rational)
    # PC2 = Self-Expression (Survival vs Self-Expression)
    lang_groups = llm.groupby("language")
    lang_stats = {}
    for lang, g in lang_groups:
        lang_stats[lang] = {
            "n": len(g),
            "secular_rational_mean": fmt(g["PC1_rescaled"].mean()),
            "secular_rational_std": fmt(g["PC1_rescaled"].std()),
            "self_expression_mean": fmt(g["PC2_rescaled"].mean()),
            "self_expression_std": fmt(g["PC2_rescaled"].std()),
        }

    print("\n  Per-language Secular-Rational (PC1) means:")
    for lang in sorted(lang_stats.keys()):
        s = lang_stats[lang]
        print(f"    {lang:8s}: mean={s['secular_rational_mean']:.3f}, std={s['secular_rational_std']:.3f}, n={s['n']}")

    # One-way ANOVA on PC1 (Secular-Rational) across languages
    pc1_groups = [g["PC1_rescaled"].values for _, g in lang_groups]
    if len(pc1_groups) >= 2:
        f_sr, p_sr = stats.f_oneway(*pc1_groups)
    else:
        f_sr, p_sr = np.nan, np.nan

    # One-way ANOVA on PC2 (Self-Expression) across languages
    pc2_groups = [g["PC2_rescaled"].values for _, g in lang_groups]
    if len(pc2_groups) >= 2:
        f_se, p_se = stats.f_oneway(*pc2_groups)
    else:
        f_se, p_se = np.nan, np.nan

    # Divergence between English and Arabic on Secular-Rational
    en_sr = llm[llm["language"] == "en"]["PC1_rescaled"].mean()
    ar_sr = llm[llm["language"] == "ar"]["PC1_rescaled"].mean()
    divergence = en_sr - ar_sr

    print(f"\n  ANOVA Secular-Rational (PC1): F={f_sr:.3f}, p={p_sr:.4f}")
    print(f"  ANOVA Self-Expression (PC2):  F={f_se:.3f}, p={p_se:.4f}")
    print(f"  English-Arabic divergence on Secular-Rational: {divergence:.2f}")

    result = {
        "study": "Study 1: Pervasive Secular-Rational Bias",
        "n_models": n_models,
        "models": models,
        "n_combinations": n_combinations,
        "language_stats": lang_stats,
        "anova_secular_rational_pc1": {
            "F": fmt(f_sr), "p": fmt(p_sr, 4),
            "significant": bool(p_sr < 0.05) if not np.isnan(p_sr) else None,
        },
        "anova_self_expression_pc2": {
            "F": fmt(f_se), "p": fmt(p_se, 4),
            "significant": bool(p_se < 0.05) if not np.isnan(p_se) else None,
        },
        "en_ar_divergence_secular_rational": fmt(divergence),
    }

    save_json(result, "study1_intrinsic_bias.json")
    return result


# ===========================================================================
# Study 2: Language Divergence (English Advantage)
# ===========================================================================

def compute_english_advantage_per_model(rp_df: pd.DataFrame, ivs_coords: dict) -> pd.DataFrame:
    """Per-model paired comparison for non-English-native countries.

    Native language is any language that is NOT 'en' or 'en-native'.
    For French-speaking countries (France, Haiti, Mali, Burkina Faso), 'fr' IS the native language.
    For Arabic countries, 'ar' is the native language (fr is NOT native for them).
    """
    llm = rp_df[rp_df["data_source"] != "IVS"].copy()
    llm = llm[~llm["model_name"].isin(EXCLUDED_ROLEPLAY)]
    llm = llm[~llm["Country"].isin(EN_NATIVE_COUNTRIES)]

    # English data: use 'en' where available, fall back to 'en-native'
    # (e.g. Hong Kong has en-native but not en, since it was previously
    # classified as EN_NATIVE and interviewed with en-native tag)
    en_df = llm[llm["language"].isin(["en", "en-native"])].copy()
    # Prefer 'en' over 'en-native': drop en-native rows for countries that have 'en'
    countries_with_en = set(en_df[en_df["language"] == "en"]["Country"].unique())
    en_df = en_df[~((en_df["language"] == "en-native") & en_df["Country"].isin(countries_with_en))]

    # Native = everything except en and en-native.
    # For Arabic countries that also have fr data, we pick only 'ar' as native.
    # For French-speaking countries (France, Haiti, Mali, Burkina Faso), 'fr' is native.
    FRENCH_NATIVE_COUNTRIES = {"France", "Haiti", "Mali", "Burkina Faso",
                               "Belgium", "Luxembourg", "Switzerland"}
    native_df = llm[~llm["language"].isin(["en", "en-native"])].copy()

    # For Arabic countries, keep only 'ar' as native (exclude 'fr')
    arabic_mask = native_df["Country"].isin(ARABIC_COUNTRIES_12) & (native_df["language"] == "fr")
    native_df = native_df[~arabic_mask]

    rows = []
    for country in native_df["Country"].dropna().unique():
        matched = match_country(country, ivs_coords)
        if matched is None:
            continue
        ivs = ivs_coords[matched]

        c_en = en_df[en_df["Country"] == country]
        c_nat = native_df[native_df["Country"] == country]

        for native_lang in c_nat["language"].unique():
            nat_l = c_nat[c_nat["language"] == native_lang]
            for model in nat_l["model_name"].unique():
                nat_m = nat_l[nat_l["model_name"] == model]
                en_m = c_en[c_en["model_name"] == model]
                if nat_m.empty or en_m.empty:
                    continue

                d_native = euclidean(
                    nat_m["PC1_rescaled"].mean(), nat_m["PC2_rescaled"].mean(),
                    ivs["PC1"], ivs["PC2"],
                )
                d_en = euclidean(
                    en_m["PC1_rescaled"].mean(), en_m["PC2_rescaled"].mean(),
                    ivs["PC1"], ivs["PC2"],
                )

                ea_pct = (d_native - d_en) / d_native * 100 if d_native > 0 else 0.0
                log_ratio = float(np.log(d_en / d_native)) if d_native > 0 and d_en > 0 else np.nan

                rows.append({
                    "country": matched,
                    "native_language": native_lang,
                    "model_name": model,
                    "cultural_region": ivs["cultural_region"],
                    "is_islamic": ivs["is_islamic"],
                    "d_native": d_native,
                    "d_english": d_en,
                    "english_advantage_pct": ea_pct,
                    "log_ratio": log_ratio,
                    "d_diff": d_native - d_en,
                })

    return pd.DataFrame(rows)


def run_study2(ea_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("STUDY 2: Language Divergence (English Advantage)")
    print("=" * 70)

    n_models = ea_df["model_name"].nunique()
    n_countries = ea_df["country"].nunique()
    n_observations = len(ea_df)
    print(f"  Models: {n_models}, Non-English countries: {n_countries}")
    print(f"  Total model×country×language observations: {n_observations}")

    # Paper's method: average distances per country across all model-language pairs,
    # then compute EA per country, then do paired t-test at the country level.
    country_agg = ea_df.groupby("country").agg(
        mean_d_native=("d_native", "mean"),
        mean_d_english=("d_english", "mean"),
    ).reset_index()
    country_agg["ea_from_means"] = (
        (country_agg["mean_d_native"] - country_agg["mean_d_english"])
        / country_agg["mean_d_native"] * 100
    )

    # Overall EA from grand mean of distances
    overall_ea = (ea_df["d_native"].mean() - ea_df["d_english"].mean()) / ea_df["d_native"].mean() * 100

    # Paired t-test at country level (n = n_countries)
    country_diff = country_agg["mean_d_native"].values - country_agg["mean_d_english"].values
    t_country, p_country = stats.ttest_rel(
        country_agg["mean_d_native"].values, country_agg["mean_d_english"].values
    )
    d_cohen_country = cohens_d_paired(country_diff)

    print(f"  Overall English Advantage: {overall_ea:+.1f}%")
    print(f"  Paired t-test (country-level, n={n_countries}): "
          f"t={t_country:.3f}, p={p_country:.6f}, Cohen's d={d_cohen_country:.3f}")

    # Per-country summary
    country_summary = ea_df.groupby("country").agg(
        n_models=("model_name", "nunique"),
        mean_d_native=("d_native", "mean"),
        mean_d_english=("d_english", "mean"),
        mean_ea_pct=("english_advantage_pct", "mean"),
        median_ea_pct=("english_advantage_pct", "median"),
    ).reset_index()
    country_summary["ea_from_means"] = (
        (country_summary["mean_d_native"] - country_summary["mean_d_english"])
        / country_summary["mean_d_native"] * 100
    )
    country_summary = country_summary.sort_values("ea_from_means", ascending=False)

    dist_stats = {
        "mean": fmt(country_agg["ea_from_means"].mean()),
        "median": fmt(country_agg["ea_from_means"].median()),
        "std": fmt(country_agg["ea_from_means"].std()),
        "min_country": country_agg.loc[country_agg["ea_from_means"].idxmin(), "country"],
        "min_value": fmt(country_agg["ea_from_means"].min()),
        "max_country": country_agg.loc[country_agg["ea_from_means"].idxmax(), "country"],
        "max_value": fmt(country_agg["ea_from_means"].max()),
    }

    print(f"\n  Distribution of per-country mean EA:")
    print(f"    Mean={dist_stats['mean']:.1f}%, Median={dist_stats['median']:.1f}%")
    print(f"    Max native advantage: {dist_stats['min_country']} ({dist_stats['min_value']:.1f}%)")
    print(f"    Max English advantage: {dist_stats['max_country']} ({dist_stats['max_value']:.1f}%)")

    result = {
        "study": "Study 2: Language Divergence",
        "n_models": n_models,
        "n_countries": n_countries,
        "n_observations": n_observations,
        "overall_english_advantage_pct": fmt(overall_ea),
        "paired_t_test_country_level": {
            "n": n_countries,
            "t": fmt(t_country), "p": fmt(p_country, 6),
            "cohens_d": fmt(d_cohen_country),
            "significant": bool(p_country < 0.05),
        },
        "distribution": dist_stats,
        "per_country": country_summary.to_dict(orient="records"),
    }

    save_json(result, "study2_english_advantage.json")
    return result


# ===========================================================================
# Study 3: Digital Orientalism
# ===========================================================================

def run_study3(ea_df: pd.DataFrame, rp_df: pd.DataFrame, ivs_coords: dict) -> dict:
    print("\n" + "=" * 70)
    print("STUDY 3: Digital Orientalism")
    print("=" * 70)

    results = {}

    # --- 3a. Regional analysis ---
    print("\n  --- 3a. Regional English Advantage ---")
    region_results = {}
    for region, g in ea_df.groupby("cultural_region"):
        if not region or pd.isna(region):
            continue

        # Country-aggregated EA (for reporting mean EA per region)
        country_agg = g.groupby("country").agg(
            mean_d_native=("d_native", "mean"),
            mean_d_english=("d_english", "mean"),
        ).reset_index()
        country_agg["ea_from_means"] = (
            (country_agg["mean_d_native"] - country_agg["mean_d_english"])
            / country_agg["mean_d_native"] * 100
        )
        ea_vals = country_agg["ea_from_means"]
        n_countries = len(ea_vals)
        n_models = g["model_name"].nunique()

        # Row-level paired t-test (as in the paper): tests d_native vs d_english
        t_r, p_r = stats.ttest_rel(g["d_native"].values, g["d_english"].values)

        # Count models with positive EA
        model_ea_agg = g.groupby("model_name").apply(
            lambda mg: (mg["d_native"].mean() - mg["d_english"].mean()) / mg["d_native"].mean() * 100
            if mg["d_native"].mean() > 0 else 0,
            include_groups=False,
        )
        n_positive_models = int((model_ea_agg > 0).sum())

        region_results[region] = {
            "n_countries": n_countries,
            "n_models": n_models,
            "n_observations": len(g),
            "mean_ea_pct": fmt(ea_vals.mean()),
            "std_ea_pct": fmt(ea_vals.std()),
            "t_test_paired": {
                "t": fmt(t_r), "p": fmt(p_r, 6),
                "significant": bool(p_r < 0.05) if not np.isnan(p_r) else None,
            },
            "n_models_positive_ea": n_positive_models,
            "n_models_total": int(len(model_ea_agg)),
        }
        print(f"    {region:25s}: mean={ea_vals.mean():+.1f}%, n_countries={n_countries}, "
              f"paired t={t_r:.2f}, p={p_r:.6f}, {n_positive_models}/{len(model_ea_agg)} models positive")

    results["regional_analysis"] = region_results

    # Western vs Non-Western contrast
    western_regions = {"Catholic Europe", "Protestant Europe"}
    non_western_regions = {"African-Islamic", "Confucian", "Latin America", "Orthodox Europe"}

    western = country_ea_paper_method(ea_df[ea_df["cultural_region"].isin(western_regions)])
    non_western = country_ea_paper_method(ea_df[ea_df["cultural_region"].isin(non_western_regions)])

    if len(western) >= 2 and len(non_western) >= 2:
        t_wn, p_wn = stats.ttest_ind(non_western, western)
        d_wn = cohens_d(non_western.values, western.values)
    else:
        t_wn, p_wn, d_wn = np.nan, np.nan, np.nan

    results["western_vs_nonwestern"] = {
        "western_mean": fmt(western.mean()),
        "nonwestern_mean": fmt(non_western.mean()),
        "t_test": {"t": fmt(t_wn), "p": fmt(p_wn, 6), "cohens_d": fmt(d_wn)},
    }
    print(f"\n    Western vs Non-Western: d={d_wn:.2f}, t={t_wn:.2f}, p={p_wn:.4f}")

    # --- 3b. French Advantage ---
    print("\n  --- 3b. French Advantage (12 Arabic Countries) ---")
    french_result = compute_french_advantage(rp_df, ivs_coords)
    results["french_advantage"] = french_result

    # --- 3c. Model Origin Effects ---
    print("\n  --- 3c. Model Origin Effects ---")
    ea_df_with_origin = ea_df.copy()
    ea_df_with_origin["model_origin"] = ea_df_with_origin["model_name"].map(MODEL_ORIGIN)

    western_regions = {"Catholic Europe", "Protestant Europe"}
    non_western_regions = {"African-Islamic", "Confucian", "Latin America", "Orthodox Europe"}

    origin_results = {}
    for origin, g in ea_df_with_origin.groupby("model_origin"):
        if not origin:
            continue
        by_region = {}
        for region, rg in g.groupby("cultural_region"):
            if not region:
                continue
            reg_ea = country_ea_paper_method(rg)
            by_region[region] = fmt(reg_ea.mean())

        # Western vs Non-Western within this origin group
        w = country_ea_paper_method(g[g["cultural_region"].isin(western_regions)])
        nw = country_ea_paper_method(g[g["cultural_region"].isin(non_western_regions)])
        if len(w) >= 2 and len(nw) >= 2:
            t_o, p_o = stats.ttest_ind(nw, w)
            d_o = cohens_d(nw.values, w.values)
        else:
            t_o, p_o, d_o = np.nan, np.nan, np.nan

        origin_results[origin] = {
            "n_models": g["model_name"].nunique(),
            "models": sorted(g["model_name"].unique()),
            "by_region": by_region,
            "western_vs_nonwestern": {
                "t": fmt(t_o), "p": fmt(p_o, 6), "cohens_d": fmt(d_o),
            },
        }
        print(f"    {origin:8s} (n={g['model_name'].nunique()}): "
              + ", ".join(f"{r}={v:+.1f}%" for r, v in sorted(by_region.items()))
              + f"  | W vs NW: t={t_o:.2f}, p={p_o:.4f}")

    results["model_origin"] = origin_results

    # --- 3d. Cultural distance correlation ---
    print("\n  --- 3d. Cultural Distance Correlation ---")
    prot_europe_countries = [c for c, v in ivs_coords.items() if v["cultural_region"] == "Protestant Europe"]
    if prot_europe_countries:
        pe_pc1 = np.mean([ivs_coords[c]["PC1"] for c in prot_europe_countries])
        pe_pc2 = np.mean([ivs_coords[c]["PC2"] for c in prot_europe_countries])

        # Compute cultural distance for each country
        cult_dist = {}
        for c in ea_df["country"].unique():
            m = match_country(c, ivs_coords)
            if m:
                cult_dist[c] = euclidean(ivs_coords[m]["PC1"], ivs_coords[m]["PC2"], pe_pc1, pe_pc2)

        # Row-level correlation (each observation = one model × country × language pair)
        ea_with_cd = ea_df.copy()
        ea_with_cd["cultural_distance"] = ea_with_cd["country"].map(cult_dist)
        valid = ea_with_cd.dropna(subset=["cultural_distance"])

        if len(valid) >= 5:
            r_pearson, p_pearson = stats.pearsonr(valid["cultural_distance"], valid["english_advantage_pct"])
            r_spearman, p_spearman = stats.spearmanr(valid["cultural_distance"], valid["english_advantage_pct"])
        else:
            r_pearson, p_pearson, r_spearman, p_spearman = np.nan, np.nan, np.nan, np.nan

        # Also compute country-level for reference
        country_ea = country_ea_paper_method(ea_df)
        common = sorted(set(country_ea.index) & set(cult_dist.keys()))
        if len(common) >= 5:
            x_c = np.array([cult_dist[c] for c in common])
            y_c = np.array([country_ea[c] for c in common])
            r_country, p_country = stats.pearsonr(x_c, y_c)
        else:
            r_country, p_country = np.nan, np.nan

        results["cultural_distance_correlation"] = {
            "row_level": {
                "pearson_r": fmt(r_pearson), "pearson_p": fmt(p_pearson, 4),
                "spearman_rho": fmt(r_spearman), "spearman_p": fmt(p_spearman, 4),
                "n": len(valid),
            },
            "country_level": {
                "pearson_r": fmt(r_country), "pearson_p": fmt(p_country, 4),
                "n": len(common),
            },
        }
        print(f"    Row-level (n={len(valid)}): Pearson r={r_pearson:.3f}, p={p_pearson:.4f}")
        print(f"    Row-level: Spearman rho={r_spearman:.3f}, p={p_spearman:.4f}")
        print(f"    Country-level (n={len(common)}): Pearson r={r_country:.3f}, p={p_country:.4f}")

    # --- 3e. Language family analysis ---
    # Paper uses row-level (per-observation) EA mean, not country-aggregated.
    print("\n  --- 3e. Language Family Analysis ---")
    ea_with_family = ea_df.copy()
    ea_with_family["language_family"] = ea_with_family["native_language"].map(LANGUAGE_FAMILY)

    family_results = {}
    for family, g in ea_with_family.groupby("language_family"):
        if not family:
            continue
        n_countries = g["country"].nunique()
        n_obs = len(g)
        row_ea = g["english_advantage_pct"].values

        # Row-level t-test (paper's approach)
        if len(row_ea) >= 2:
            t_row, p_row = stats.ttest_1samp(row_ea, 0)
        else:
            t_row, p_row = np.nan, np.nan

        # Country-aggregated for reference
        country_ea = country_ea_paper_method(g)
        if len(country_ea) >= 2:
            t_country, p_country = stats.ttest_1samp(country_ea, 0)
        else:
            t_country, p_country = np.nan, np.nan

        family_results[family] = {
            "n_countries": n_countries,
            "n_observations": n_obs,
            "row_level_mean_ea_pct": fmt(np.mean(row_ea)),
            "country_level_mean_ea_pct": fmt(country_ea.mean()),
            "row_level_t_test": {"t": fmt(t_row), "p": fmt(p_row, 4)},
            "country_level_t_test": {"t": fmt(t_country), "p": fmt(p_country, 4)},
        }
        print(f"    {family:12s}: row_mean={np.mean(row_ea):+.1f}% (n={n_obs}), "
              f"country_mean={country_ea.mean():+.1f}% (n={n_countries}), "
              f"row_p={p_row:.4f}, country_p={p_country:.4f}")

    results["language_family"] = family_results

    save_json(results, "study3_digital_orientalism.json")
    return results


def compute_french_advantage(rp_df: pd.DataFrame, ivs_coords: dict) -> dict:
    """
    French advantage for 12 Arabic-speaking countries (Study 3).

    French Advantage (%) = (d_arabic - d_french) / d_arabic * 100
      Positive => French roleplay is closer to IVS than Arabic.
      Negative => Arabic is closer than French.

    Arabic and French must use the same PCA pipeline: coordinates from
    ``roleplay_ml_pca_entity_scores_latest.pkl`` only. Re-projecting French from
    raw JSON alone desynchronises French vs Arabic and biases distances.

    Phi-3 excluded from French side (paper: no reliable French). Pairwise % can
    explode when d_arabic is tiny (same issue as English Advantage); we also
    report median FA%, winsorized mean, and country-level mean FA%.
    """
    llm = rp_df[rp_df["data_source"] != "IVS"].copy()
    llm = llm[~llm["model_name"].isin(EXCLUDED_ROLEPLAY)]

    ar_pca = llm[(llm["language"] == "ar") & (llm["Country"].isin(ARABIC_COUNTRIES_12))]
    fr_pca = llm[(llm["language"] == "fr") & (llm["Country"].isin(ARABIC_COUNTRIES_12))]
    fr_pca = fr_pca[~fr_pca["model_name"].str.contains("phi-3", case=False, na=False)]

    if fr_pca.empty:
        print("    No French roleplay PCA rows in unified pkl.")
        return {"error": "no_french_data"}

    rows = []
    for country in ARABIC_COUNTRIES_12:
        if country not in ivs_coords:
            continue
        ivs = ivs_coords[country]
        ar_c = ar_pca[ar_pca["Country"] == country]
        fr_c = fr_pca[fr_pca["Country"] == country]

        for model in ar_c["model_name"].unique():
            ar_m = ar_c[ar_c["model_name"] == model]
            fr_m = fr_c[fr_c["model_name"] == model]
            if ar_m.empty or fr_m.empty:
                continue

            ar_pc1 = float(ar_m["PC1_rescaled"].mean())
            ar_pc2 = float(ar_m["PC2_rescaled"].mean())
            fr_pc1 = float(fr_m["PC1_rescaled"].mean())
            fr_pc2 = float(fr_m["PC2_rescaled"].mean())

            d_ar = euclidean(ar_pc1, ar_pc2, ivs["PC1"], ivs["PC2"])
            d_fr = euclidean(fr_pc1, fr_pc2, ivs["PC1"], ivs["PC2"])
            adv = (d_ar - d_fr) / d_ar * 100 if d_ar > 0 else 0.0

            rows.append({
                "country": country,
                "model_name": model,
                "d_arabic": d_ar,
                "d_french": d_fr,
                "french_advantage_pct": adv,
                "ar_pc1": ar_pc1,
                "ar_pc2": ar_pc2,
                "fr_pc1": fr_pc1,
                "fr_pc2": fr_pc2,
            })

    if not rows:
        return {"error": "no_paired_data"}

    fa_df = pd.DataFrame(rows)

    t_fa, p_fa = stats.ttest_rel(fa_df["d_arabic"], fa_df["d_french"])
    diff = fa_df["d_arabic"].values - fa_df["d_french"].values
    d_fa = cohens_d_paired(diff)

    overall = (fa_df["d_arabic"].mean() - fa_df["d_french"].mean()) / fa_df["d_arabic"].mean() * 100
    median_fa = float(np.median(fa_df["french_advantage_pct"].values))
    mean_fa_winsor = float(np.mean(np.clip(fa_df["french_advantage_pct"].values, -200.0, 200.0)))

    country_level_fas = []
    for _, g in fa_df.groupby("country"):
        md_ar = g["d_arabic"].mean()
        md_fr = g["d_french"].mean()
        if md_ar > 0:
            country_level_fas.append((md_ar - md_fr) / md_ar * 100)
    country_level_overall = float(np.mean(country_level_fas)) if country_level_fas else float("nan")

    n_models_fa = fa_df["model_name"].nunique()
    n_countries_fa = fa_df["country"].nunique()
    n_negative = int((fa_df["french_advantage_pct"] < 0).sum())

    print(f"    French Advantage (unified PCA): {overall:+.1f}%, n_models={n_models_fa}, "
          f"n_countries={n_countries_fa}, n_pairs={len(fa_df)}")
    print(f"    Paired t-test (distances): t={t_fa:.3f}, p={p_fa:.4f}, Cohen's d={d_fa:.3f}")
    print(f"    Median FA%: {median_fa:+.1f}; winsor mean [-200,200]: {mean_fa_winsor:+.1f}; "
          f"country-level mean: {country_level_overall:+.1f}% ({n_negative}/{len(fa_df)} pairs < 0)")

    per_country = {}
    for c, g in fa_df.groupby("country"):
        c_adv = (g["d_arabic"].mean() - g["d_french"].mean()) / g["d_arabic"].mean() * 100
        per_country[c] = fmt(c_adv)

    fa_path = OUT_DIR / "french_advantage_12_arab_countries.csv"
    fa_df.to_csv(fa_path, index=False)
    # Legacy path expected by docs / paper_data README
    fa_df.to_csv(PAPER_DATA_DIR / "french_advantage_12_countries.csv", index=False)
    fa_df.groupby("model_name")["french_advantage_pct"].agg(["mean", "count"]).reset_index().sort_values(
        "mean"
    ).to_csv(OUT_DIR / "french_advantage_by_model.csv", index=False)
    fa_df.groupby("country")["french_advantage_pct"].agg(["mean", "count"]).reset_index().sort_values(
        "mean"
    ).to_csv(OUT_DIR / "french_advantage_by_country.csv", index=False)

    return {
        "overall_french_advantage_pct": fmt(overall),
        "median_french_advantage_pct_pairwise": fmt(median_fa),
        "mean_french_advantage_pct_winsorized_200": fmt(mean_fa_winsor),
        "country_level_mean_french_advantage_pct": fmt(country_level_overall),
        "n_models": n_models_fa,
        "n_countries": n_countries_fa,
        "n_observations": len(fa_df),
        "n_pairs_negative_fa": n_negative,
        "paired_t_test": {
            "t": fmt(t_fa), "p": fmt(p_fa, 4),
            "cohens_d": fmt(d_fa),
            "significant": bool(p_fa < 0.05),
        },
        "per_country": per_country,
    }


# ===========================================================================
# Study 4: Colonial Legacies
# ===========================================================================

def run_study4(ea_df: pd.DataFrame, rp_df: pd.DataFrame, ivs_coords: dict) -> dict:
    print("\n" + "=" * 70)
    print("STUDY 4: Colonial Legacies")
    print("=" * 70)

    results = {}

    # --- 4a. Confucian intra-regional stratification ---
    print("\n  --- 4a. Confucian Intra-Regional Stratification ---")
    CONFUCIAN_DISPLAY = {
        "Korea, Republic of": "South Korea",
        "Taiwan, Province of China": "Taiwan",
    }

    # For all Confucian entities, compute EA using en/en-native vs native language
    llm_all = rp_df[(rp_df["data_source"] != "IVS") & (~rp_df["model_name"].isin(EXCLUDED_ROLEPLAY))]

    def compute_ea_for_entity(entity_name, ivs_c):
        """Compute EA for a single entity, handling both en and en-native."""
        if entity_name not in ivs_c:
            return None, 0
        ivs = ivs_c[entity_name]
        e_all = llm_all[llm_all["Country"] == entity_name]

        # English data: prefer 'en', fall back to 'en-native'
        en_data = e_all[e_all["language"] == "en"]
        if en_data.empty:
            en_data = e_all[e_all["language"] == "en-native"]

        # Native data: non-English, non-en-native, non-fr-for-arabic
        native_data = e_all[~e_all["language"].isin(["en", "en-native", "fr"])]
        if native_data.empty:
            native_data = e_all[e_all["language"] == "fr"]  # for French-native countries

        if en_data.empty or native_data.empty:
            return None, 0

        d_en_list, d_nat_list = [], []
        for model in native_data["model_name"].unique():
            en_m = en_data[en_data["model_name"] == model]
            nat_m = native_data[native_data["model_name"] == model]
            if en_m.empty or nat_m.empty:
                continue
            d_en = euclidean(en_m["PC1_rescaled"].mean(), en_m["PC2_rescaled"].mean(), ivs["PC1"], ivs["PC2"])
            d_nat = euclidean(nat_m["PC1_rescaled"].mean(), nat_m["PC2_rescaled"].mean(), ivs["PC1"], ivs["PC2"])
            d_en_list.append(d_en)
            d_nat_list.append(d_nat)

        if not d_en_list:
            return None, 0
        mean_d_nat = np.mean(d_nat_list)
        mean_d_en = np.mean(d_en_list)
        ea = (mean_d_nat - mean_d_en) / mean_d_nat * 100 if mean_d_nat > 0 else 0
        return ea, len(d_en_list)

    confucian_ea = {}
    for entity in CONFUCIAN_ENTITIES:
        display = CONFUCIAN_DISPLAY.get(entity, entity)
        # First check if it's in the main ea_df
        e_data = ea_df[ea_df["country"] == entity]
        if not e_data.empty:
            ea_val = entity_ea_paper(e_data)
            n_models = e_data["model_name"].nunique()
        else:
            # Compute directly for entities not in main ea_df
            ea_val, n_models = compute_ea_for_entity(entity, ivs_coords)
        if ea_val is not None:
            confucian_ea[display] = {
                "mean_ea_pct": fmt(ea_val),
                "n_models": n_models,
            }
            print(f"    {display:15s}: EA={ea_val:+.1f}%, n_models={n_models}")

    results["confucian_stratification"] = confucian_ea

    # --- 4b. Sub-Saharan Africa: British colonies ---
    print("\n  --- 4b. British Colonies in Sub-Saharan Africa ---")
    llm = rp_df[(rp_df["data_source"] != "IVS") & (~rp_df["model_name"].isin(EXCLUDED_ROLEPLAY))]
    en_data = llm[llm["language"].isin(["en", "en-native"])]

    africa_british_results = {}
    pc2_biases = []
    for country in AFRICA_BRITISH:
        c_data = en_data[en_data["Country"] == country]
        if c_data.empty or country not in ivs_coords:
            continue
        ivs_pc2 = ivs_coords[country]["PC2"]
        model_biases = []
        for model, mg in c_data.groupby("model_name"):
            bias = mg["PC2_rescaled"].mean() - ivs_pc2
            model_biases.append(bias)
        mean_bias = np.mean(model_biases)
        pc2_biases.append(mean_bias)
        africa_british_results[country] = {"mean_pc2_bias": fmt(mean_bias), "n_models": len(model_biases)}
        print(f"    {country:20s}: PC2 bias={mean_bias:+.2f}")

    if len(pc2_biases) >= 2:
        t_ab, p_ab = stats.ttest_1samp(pc2_biases, 0)
    else:
        t_ab, p_ab = np.nan, np.nan

    aggregate_bias = np.mean(pc2_biases) if pc2_biases else np.nan
    print(f"    Aggregate: mean={aggregate_bias:+.2f}, t={t_ab:.2f}, p={p_ab:.4f}")

    results["africa_british"] = {
        "per_country": africa_british_results,
        "aggregate_mean_bias": fmt(aggregate_bias),
        "t_test": {"t": fmt(t_ab), "p": fmt(p_ab, 4), "significant": bool(p_ab < 0.05) if not np.isnan(p_ab) else None},
    }

    # --- 4c. Sub-Saharan Africa: French colonies ---
    print("\n  --- 4c. French Colonies in Sub-Saharan Africa ---")
    africa_french_results = {}
    for country in AFRICA_FRENCH:
        c_data = ea_df[ea_df["country"] == country]
        if not c_data.empty:
            ea_val = entity_ea_paper(c_data)
            africa_french_results[country] = {
                "mean_ea_pct": fmt(ea_val),
                "n_models": c_data["model_name"].nunique(),
            }
            print(f"    {country:20s}: EA={ea_val:+.1f}%")

    french_colony_ea = ea_df[ea_df["country"].isin(AFRICA_FRENCH)]
    if len(french_colony_ea) >= 2:
        country_means = country_ea_paper_method(french_colony_ea)
        if len(country_means) >= 2:
            t_fc, p_fc = stats.ttest_1samp(country_means, 0)
        else:
            t_fc, p_fc = np.nan, np.nan
        overall_fc = country_means.mean()
    else:
        t_fc, p_fc, overall_fc = np.nan, np.nan, np.nan

    results["africa_french"] = {
        "per_country": africa_french_results,
        "aggregate_mean_ea_pct": fmt(overall_fc),
        "t_test": {"t": fmt(t_fc), "p": fmt(p_fc, 4)},
    }

    # --- 4d. Latin America ---
    print("\n  --- 4d. Latin America ---")
    latam_results = {}

    # Spanish colonies
    spanish_ea = ea_df[ea_df["country"].isin(LATAM_SPANISH)]
    spanish_by_country = {}
    for c in LATAM_SPANISH:
        c_data = ea_df[ea_df["country"] == c]
        if not c_data.empty:
            ea_val = entity_ea_paper(c_data)
            spanish_by_country[c] = fmt(ea_val)
            print(f"    {c:20s} (ES): EA={ea_val:+.1f}%")

    if not spanish_ea.empty:
        sp_country_ea = country_ea_paper_method(spanish_ea)
        latam_results["spanish_colonies"] = {
            "n": len(sp_country_ea),
            "mean_ea_pct": fmt(sp_country_ea.mean()),
            "per_country": spanish_by_country,
        }

    # Portuguese (Brazil)
    brazil_ea = ea_df[ea_df["country"] == "Brazil"]
    if not brazil_ea.empty:
        ea_val = entity_ea_paper(brazil_ea)
        latam_results["brazil"] = {"mean_ea_pct": fmt(ea_val)}
        print(f"    Brazil (PT): EA={ea_val:+.1f}%")

    # French (Haiti)
    haiti_ea = ea_df[ea_df["country"] == "Haiti"]
    if not haiti_ea.empty:
        ea_val = entity_ea_paper(haiti_ea)
        latam_results["haiti"] = {"mean_ea_pct": fmt(ea_val)}
        print(f"    Haiti (FR): EA={ea_val:+.1f}%")

    results["latin_america"] = latam_results

    save_json(results, "study4_colonial_legacies.json")
    return results


# ===========================================================================
# Regression data generation
# ===========================================================================

def generate_regression_data(ea_df: pd.DataFrame, ivs_coords: dict) -> pd.DataFrame:
    """Generate the publication-ready regression table.

    The released paper table is written to `results/paper_data/regression_data.csv`.
    A separate external covariate table is used only to supply country-level
    metadata that are not stored in the final paper table itself.
    """
    print("\n" + "=" * 70)
    print("REGRESSION DATA GENERATION (v5)")
    print("=" * 70)

    # Load the upstream covariate lookup table. Prefer the stable public name,
    # but accept older local paths for backwards compatibility.
    candidate_covariate_paths = [
        PROJECT_ROOT / "data" / "external" / "regression_covariates.csv",
    ]
    meta_cols = [
        "country", "internet_collectivity_index", "colonial_power",
        "colonial_history", "gdp_per_capita", "HDI", "has_colonial_history",
        "colony_CN", "colony_ES", "colony_FR", "colony_IT", "colony_JP",
        "colony_PT", "colony_RU", "colony_UK", "freedom_on_net",
    ]
    # The older covariate table uses short country names; the paper data use
    # the IVS formal names.
    covariate_name_map = {
        "South Korea": "Korea, Republic of",
        "Taiwan": "Taiwan, Province of China",
    }
    covariate_path = next((path for path in candidate_covariate_paths if path.exists()), None)
    if covariate_path is not None:
        covariates = pd.read_csv(covariate_path)
        covariates["country"] = covariates["country"].map(
            lambda x: covariate_name_map.get(x, x)
        )
        meta = covariates[meta_cols].drop_duplicates(subset=["country"])
    else:
        meta = pd.DataFrame(columns=meta_cols)

    # Add cultural_region and language_family
    ea_df = ea_df.copy()
    ea_df["language_family"] = ea_df["native_language"].map(LANGUAGE_FAMILY)
    ea_df["model_origin"] = ea_df["model_name"].map(MODEL_ORIGIN)

    # Winsorize EA to [-200, 200]
    ea_df["ea_winsorized"] = ea_df["english_advantage_pct"].clip(-200, 200)

    # Log ratio (already computed)
    # log_ratio > 0 means English is farther (native advantage)
    # log_ratio < 0 means English is closer (English advantage)

    # Merge metadata
    reg = ea_df.merge(meta, on="country", how="left")

    # Reorder columns
    col_order = [
        "model_name", "country", "native_language", "cultural_region", "is_islamic",
        "language_family", "model_origin",
        "d_english", "d_native", "d_diff",
        "english_advantage_pct", "ea_winsorized", "log_ratio",
        "internet_collectivity_index", "colonial_power", "colonial_history",
        "gdp_per_capita", "HDI", "has_colonial_history",
        "colony_CN", "colony_ES", "colony_FR", "colony_IT",
        "colony_JP", "colony_PT", "colony_RU", "colony_UK", "freedom_on_net",
    ]
    existing_cols = [c for c in col_order if c in reg.columns]
    reg = reg[existing_cols]

    out_path = OUT_DIR / "regression_data_v5.csv"
    reg.to_csv(out_path, index=False)

    print(f"  Rows: {len(reg)}")
    print(f"  Models: {reg['model_name'].nunique()}")
    print(f"  Countries: {reg['country'].nunique()}")
    print(f"  Saved: {out_path}")

    # Stats on extreme values
    n_extreme = (reg["english_advantage_pct"].abs() > 100).sum()
    print(f"\n  Extreme values (|EA| > 100%): {n_extreme} rows")
    print(f"  After winsorize: min={reg['ea_winsorized'].min():.1f}%, max={reg['ea_winsorized'].max():.1f}%")
    print(f"  Log ratio: min={reg['log_ratio'].min():.3f}, max={reg['log_ratio'].max():.3f}")
    print(f"  Log ratio mean={reg['log_ratio'].mean():.3f}, median={reg['log_ratio'].median():.3f}")

    return reg


# ===========================================================================
# Model Imitation Accuracy (supplements Study 2)
# ===========================================================================

# Open-source vs closed-source classification
OPEN_SOURCE_MODELS = {
    "llama-3.2-3b-instruct", "llama-3.3-70b-instruct",
    "mistral-nemo", "phi-3-mini-128k-instruct", "gemma-3-4b-it",
    "deepseek-chat", "deepseek-chat-v3.1", "qwen3-max", "kimi-k2",
}
CLOSED_SOURCE_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-5.1",
    "claude-3-7-sonnet-20250219", "claude-sonnet-4.5",
    "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview",
    "mistral-medium-3.1", "doubao-1-5-pro-32k-250115", "grok-4.1-fast",
}
MODEL_VENDOR = {
    "gpt-4o": "OpenAI", "gpt-4o-mini": "OpenAI", "gpt-5.1": "OpenAI",
    "claude-3-7-sonnet-20250219": "Anthropic", "claude-sonnet-4.5": "Anthropic",
    "gemini-2.5-flash": "Google", "gemini-2.5-pro": "Google",
    "gemini-3-pro-preview": "Google", "gemma-3-4b-it": "Google",
    "llama-3.2-3b-instruct": "Meta", "llama-3.3-70b-instruct": "Meta",
    "mistral-medium-3.1": "Mistral", "mistral-nemo": "Mistral",
    "deepseek-chat": "DeepSeek", "deepseek-chat-v3.1": "DeepSeek",
    "qwen3-max": "Alibaba", "doubao-1-5-pro-32k-250115": "ByteDance",
    "kimi-k2": "Moonshot", "grok-4.1-fast": "xAI",
    "phi-3-mini-128k-instruct": "Microsoft",
}
SMALL_MODELS = {
    "llama-3.2-3b-instruct", "phi-3-mini-128k-instruct",
    "gemma-3-4b-it", "gpt-4o-mini", "mistral-nemo",
}


def _compute_distance_to_ivs(row, ivs_coords):
    """Euclidean distance from a single PCA row to its IVS reference."""
    c = row["Country"]
    matched = match_country(c, ivs_coords)
    if matched is None:
        return np.nan
    ivs = ivs_coords[matched]
    return euclidean(row["PC1_rescaled"], row["PC2_rescaled"],
                     ivs["PC1"], ivs["PC2"])


def run_model_imitation(rp_df: pd.DataFrame, ivs_coords: dict) -> dict:
    """Model-level imitation accuracy analysis (English roleplay).

    For every model, compute the mean Euclidean distance between its English
    roleplay coordinates and the IVS ground truth across all 66 countries.
    Also compute bias direction (PC1/PC2 signed deviations) and compare
    open-source vs closed-source, model origin, and cultural-region patterns.

    Supplements Study 2 (Language Divergence) by characterizing model-level
    variation before examining language effects.

    Outputs:
        results/analysis/model_imitation_accuracy.json
        results/analysis/model_imitation_accuracy.csv
        results/analysis/model_imitation_by_region.csv
    """
    print("\n" + "=" * 70)
    print("MODEL IMITATION ACCURACY (supplements Study 2)")
    print("=" * 70)

    llm = rp_df[rp_df["data_source"] != "IVS"].copy()
    llm = llm[~llm["model_name"].isin(EXCLUDED_ROLEPLAY)]

    # Cultural region mapping
    cr_col = "Cultural Region" if "Cultural Region" in llm.columns else "cultural_region"
    cr_map = rp_df.drop_duplicates("Country").set_index("Country")[cr_col].to_dict()
    cr_map = {k: v for k, v in cr_map.items() if v is not None and v != ""}

    # --- English roleplay subset ---
    en_rp = llm[llm["language"].isin(["en", "en-native"])].copy()
    countries_with_en = set(en_rp[en_rp["language"] == "en"]["Country"].unique())
    en_rp = en_rp[~((en_rp["language"] == "en-native")
                     & en_rp["Country"].isin(countries_with_en))]

    n_models = en_rp["model_name"].nunique()
    n_countries = en_rp["Country"].nunique()
    print(f"  English roleplay: {len(en_rp)} rows, {n_models} models, {n_countries} countries")

    # --- Per-model metrics ---
    western_regions = {"Catholic Europe", "Protestant Europe", "English-Speaking"}

    model_rows = []
    model_region_rows = []
    for model in sorted(en_rp["model_name"].unique()):
        m = en_rp[en_rp["model_name"] == model]
        dists, pc1_biases, pc2_biases = [], [], []
        region_dists: dict[str, list] = {}

        for _, row in m.iterrows():
            c = row["Country"]
            matched = match_country(c, ivs_coords)
            if matched is None:
                continue
            ivs = ivs_coords[matched]
            d = euclidean(row["PC1_rescaled"], row["PC2_rescaled"],
                          ivs["PC1"], ivs["PC2"])
            dists.append(d)
            pc1_biases.append(row["PC1_rescaled"] - ivs["PC1"])
            pc2_biases.append(row["PC2_rescaled"] - ivs["PC2"])

            region = cr_map.get(c, "Other")
            region_dists.setdefault(region, []).append(d)

        if not dists:
            continue

        is_open = model in OPEN_SOURCE_MODELS
        origin = MODEL_ORIGIN.get(model, "?")
        vendor = MODEL_VENDOR.get(model, "?")
        is_small = model in SMALL_MODELS

        model_rows.append({
            "model_name": model,
            "vendor": vendor,
            "origin": origin,
            "open_source": is_open,
            "size_category": "Small" if is_small else "Large",
            "mean_distance": fmt(np.mean(dists)),
            "median_distance": fmt(np.median(dists)),
            "std_distance": fmt(np.std(dists)),
            "pc1_bias": fmt(np.mean(pc1_biases)),
            "pc2_bias": fmt(np.mean(pc2_biases)),
            "n_countries": m["Country"].nunique(),
        })

        for region, rd in region_dists.items():
            is_western = region in western_regions
            model_region_rows.append({
                "model_name": model,
                "cultural_region": region,
                "is_western": is_western,
                "mean_distance": fmt(np.mean(rd)),
                "n_countries": len(rd),
                "open_source": is_open,
                "origin": origin,
            })

    model_df = pd.DataFrame(model_rows).sort_values("mean_distance")
    region_df = pd.DataFrame(model_region_rows)

    # --- Print ranking ---
    print(f"\n  Model ranking (English roleplay, mean distance to IVS):")
    for i, (_, r) in enumerate(model_df.iterrows()):
        oc = "Open" if r["open_source"] else "Closed"
        print(f"    {i+1:2d}. {r['model_name']:35s} dist={r['mean_distance']:.3f}  "
              f"{oc:6s} {r['origin']:6s}  "
              f"bias=({r['pc1_bias']:+.2f}, {r['pc2_bias']:+.2f})")

    # --- Open vs Closed ---
    open_d = model_df[model_df["open_source"]]["mean_distance"]
    closed_d = model_df[~model_df["open_source"]]["mean_distance"]
    t_oc, p_oc = stats.ttest_ind(open_d, closed_d)
    pooled_std = np.sqrt((open_d.std()**2 + closed_d.std()**2) / 2)
    d_oc = (open_d.mean() - closed_d.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"\n  Open-source vs Closed-source:")
    print(f"    Open  (n={len(open_d)}): mean={open_d.mean():.3f}")
    print(f"    Closed(n={len(closed_d)}): mean={closed_d.mean():.3f}")
    print(f"    t={t_oc:.3f}, p={p_oc:.4f}, Cohen's d={d_oc:.3f}")

    # --- Origin comparison ---
    origin_stats = {}
    for origin in ["US", "China", "Europe"]:
        o_d = model_df[model_df["origin"] == origin]["mean_distance"]
        if len(o_d) > 0:
            origin_stats[origin] = {"mean": fmt(o_d.mean()), "n": len(o_d)}
            print(f"    {origin}: mean={o_d.mean():.3f} (n={len(o_d)})")

    us_d = model_df[model_df["origin"] == "US"]["mean_distance"]
    cn_d = model_df[model_df["origin"] == "China"]["mean_distance"]
    if len(us_d) > 1 and len(cn_d) > 1:
        t_uc, p_uc = stats.ttest_ind(us_d, cn_d)
        print(f"    US vs China: t={t_uc:.3f}, p={p_uc:.4f}")
        origin_stats["us_vs_china"] = {"t": fmt(t_uc), "p": fmt(p_uc, 4)}

    # --- Open/Closed × Western/Non-Western ---
    region_df["is_western_str"] = region_df["is_western"].map(
        {True: "Western", False: "Non-Western"})
    interaction = {}
    for oc_label, oc_val in [("Open", True), ("Closed", False)]:
        for wn in ["Western", "Non-Western"]:
            subset = region_df[(region_df["open_source"] == oc_val)
                               & (region_df["is_western_str"] == wn)]
            mean_d = float(subset["mean_distance"].mean()) if len(subset) > 0 else float("nan")
            interaction[f"{oc_label}_{wn}"] = fmt(mean_d)
            print(f"    {oc_label:6s} × {wn:12s}: {mean_d:.3f}")

    # --- Save outputs ---
    model_df.to_csv(OUT_DIR / "model_imitation_accuracy.csv", index=False)
    region_df.to_csv(OUT_DIR / "model_imitation_by_region.csv", index=False)

    result = {
        "study": "Study 5: Model Imitation Accuracy (English Roleplay)",
        "n_models": n_models,
        "n_countries": n_countries,
        "model_ranking": model_df.to_dict(orient="records"),
        "open_vs_closed": {
            "open_mean": fmt(open_d.mean()),
            "closed_mean": fmt(closed_d.mean()),
            "n_open": len(open_d), "n_closed": len(closed_d),
            "t": fmt(t_oc), "p": fmt(p_oc, 4), "cohens_d": fmt(d_oc),
        },
        "origin_comparison": origin_stats,
        "interaction_open_closed_x_western": interaction,
    }
    save_json(result, "model_imitation_accuracy.json")
    print(f"\n  Saved: model_imitation_accuracy.csv, model_imitation_by_region.csv")
    return result


# ===========================================================================
# Helpers
# ===========================================================================

def save_json(data, filename):
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> Saved: {path}")


def collect_all_statistics(s1, s2, s3, s4, model_imit=None) -> dict:
    """Collect all paper statistics into one JSON for easy reference."""
    all_stats = {
        "study1": s1,
        "study2": s2,
        "study3": s3,
        "study4": s4,
    }
    if model_imit is not None:
        all_stats["model_imitation"] = model_imit
    save_json(all_stats, "paper_statistics_all.json")
    return all_stats


# ===========================================================================
# Rebuild from raw interviews
# ===========================================================================

def rebuild_pca_from_raw():
    """Rebuild PCA coordinates from raw interview JSON files.

    Pipeline:
      1. Raw interview JSONs → IVS-format DataFrame (via MultilingualRoleplayDataProcessor)
      2. IVS-format + fixed PCA model → PCA coordinates (via MultilingualRoleplayPCAAnalysis)

    Uses config/country/country_codes.pkl for country metadata.
    """
    from src.roleplay_multilingual.multilingual_roleplay_data_processor import (
        MultilingualRoleplayDataProcessor,
    )
    from src.roleplay_multilingual.multilingual_roleplay_pca_analysis import (
        MultilingualRoleplayPCAAnalysis,
    )

    print("=" * 70)
    print("REBUILD PCA FROM RAW INTERVIEW DATA")
    print("=" * 70)

    # Step 1: Process raw interviews → IVS format
    print("\n--- Step 1: Processing raw interviews → IVS format ---")
    processor = MultilingualRoleplayDataProcessor(data_path=str(PROJECT_ROOT / "data"))
    result = processor.process_multilingual_data()
    ivs_df = result["ivs_df"]
    print(f"  IVS-format records: {len(ivs_df)}")
    print(f"  Countries: {ivs_df['Country'].nunique()}")
    print(f"  Models: {ivs_df['model_name'].nunique() if 'model_name' in ivs_df.columns else 'N/A'}")

    # Step 2: Project through fixed PCA
    print("\n--- Step 2: Projecting through fixed PCA model ---")
    pca_analyzer = MultilingualRoleplayPCAAnalysis(
        data_path=str(PROJECT_ROOT / "data")
    )
    entity_scores = pca_analyzer.run_analysis_with_fixed_pca()
    print(f"  Entity scores: {len(entity_scores)} rows")

    print("\n" + "=" * 70)
    print("REBUILD COMPLETE — PCA data saved to data/llm_pca/multilingual/")
    print("=" * 70)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper analysis: Study 1-4 + model imitation")
    parser.add_argument("--study", nargs="+", type=int, choices=[1, 2, 3, 4],
                        help="Run specific studies (default: all)")
    parser.add_argument("--model-imitation", action="store_true",
                        help="Run model imitation accuracy analysis (supplements Study 2)")
    parser.add_argument("--regression-only", action="store_true",
                        help="Only generate regression CSV")
    parser.add_argument("--rebuild-from-raw", action="store_true",
                        help="Rebuild PCA from raw interview JSONs before analysis")
    args = parser.parse_args()

    studies_to_run = set(args.study) if args.study else {1, 2, 3, 4}
    run_imitation = args.model_imitation or (args.study is None)

    print("=" * 70)
    print("THE VALUE ATLAS OF AI — Paper Analysis Pipeline")
    print("=" * 70)

    if args.rebuild_from_raw:
        rebuild_pca_from_raw()

    # Load data
    print("\nLoading data...")
    ivs_coords = load_ivs_coordinates()
    print(f"  IVS coordinates: {len(ivs_coords)} countries")

    intrinsic_df = load_intrinsic_pca()
    print(f"  Intrinsic PCA: {len(intrinsic_df)} entries")

    rp_df = load_roleplay_pca()
    print(f"  Roleplay PCA: {len(rp_df)} entries")

    # Compute English advantage (needed for Study 2, 3, 4 and regression)
    if studies_to_run & {2, 3, 4} or args.regression_only:
        print("\nComputing English advantage per model...")
        ea_df = compute_english_advantage_per_model(rp_df, ivs_coords)
        print(f"  English advantage entries: {len(ea_df)}, "
              f"models: {ea_df['model_name'].nunique()}, "
              f"countries: {ea_df['country'].nunique()}")

    if args.regression_only:
        generate_regression_data(ea_df, ivs_coords)
        return

    # Run studies
    s1 = s2 = s3 = s4 = None
    model_imit = None

    if 1 in studies_to_run:
        s1 = run_study1(intrinsic_df)

    if 2 in studies_to_run:
        s2 = run_study2(ea_df)

    if 3 in studies_to_run:
        s3 = run_study3(ea_df, rp_df, ivs_coords)

    if 4 in studies_to_run:
        s4 = run_study4(ea_df, rp_df, ivs_coords)

    if run_imitation:
        model_imit = run_model_imitation(rp_df, ivs_coords)

    # Generate regression data
    if studies_to_run & {2, 3, 4}:
        generate_regression_data(ea_df, ivs_coords)

    # Collect all stats
    if studies_to_run == {1, 2, 3, 4} and run_imitation:
        collect_all_statistics(s1, s2, s3, s4, model_imit)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
