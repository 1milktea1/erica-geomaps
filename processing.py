"""Data layer for the DESDR Bangladesh flood-year ranking survey geomap.

Reads a Noki survey export (Parquet), normalizes it to one row per
(respondent, rank, year), validates it, joins it to the location gazetteer and
computes location-level ranking statistics.

Interpretation rules baked in here:
- Ranks are ordinal (1 = worst flood year, 5 = fifth worst). They are not
  flood-intensity measurements, so only counts, mean/median rank and rank
  distributions are computed.
- A year absent from a respondent's top five is *unranked*, not "no flood":
  it never contributes a rank value; it only shows up as a zero count.
- Invalid rows are reported in `issues`, never corrected or imputed.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_PARQUET = HERE / "data" / "EricaTest.parquet"
GAZETTEER_CSV = HERE / "data" / "locations.csv"
BOUNDARIES_GEOJSON = HERE / "data" / "upazilas_subset.geojson"

YEAR_MIN, YEAR_MAX = 2000, 2020
RANKS = [1, 2, 3, 4, 5]
SMALL_SAMPLE_N = 5  # flag statistics built on fewer respondents than this

# Candidate source column names, checked in order (case-insensitive).
COLUMN_CANDIDATES = {
    "respondent": ["respondent", "respondent_id", "user", "user_id", "phone", "start"],
    "location": ["loc_question", "location", "loc", "region", "upazila"],
    "rank": ["rank"],
    "year": ["year"],
    "comments": ["comments", "comment"],
}
WIDE_RANK_PATTERN = re.compile(r"^rank_?([1-5])$", re.I)


# --------------------------------------------------------------------------- loading

def load_raw(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Read the Parquet file (read-only)."""
    return pd.read_parquet(path)


def inspect_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Column / dtype / null / example report for display."""
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "example": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "(all missing)" for c in df.columns],
    })


def load_gazetteer(path: str | Path = GAZETTEER_CSV) -> pd.DataFrame:
    gaz = pd.read_csv(path, keep_default_na=False)
    gaz["lat"] = gaz["lat"].astype(float)
    gaz["lon"] = gaz["lon"].astype(float)
    return gaz


def _alias_lookup(gaz: pd.DataFrame) -> dict[str, str]:
    lookup = {}
    for _, row in gaz.iterrows():
        for alias in [row["survey_name"], *str(row["aliases"]).split("|")]:
            if alias:
                lookup[_key(alias)] = row["survey_name"]
    return lookup


def _key(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _find(df: pd.DataFrame, role: str) -> str | None:
    cols = {str(c).lower(): c for c in df.columns}
    for cand in COLUMN_CANDIDATES[role]:
        if cand in cols:
            return cols[cand]
    return None


# --------------------------------------------------------------------------- normalization

def normalize(df: pd.DataFrame, gaz: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (clean, issues, info).

    clean:  respondent, location, rank (int), year (int), comment (str|NA), source_row
    issues: offending source rows with a `reason` column
    info:   which source columns were used and any notes about the input format
    """
    info: dict = {"format": None, "columns_used": {}, "notes": []}
    df = df.reset_index(drop=True).copy()
    df["source_row"] = df.index

    loc_col = _find(df, "location")
    if loc_col is None:
        raise ValueError(f"No location column found; looked for {COLUMN_CANDIDATES['location']}. "
                         f"Columns present: {list(df.columns)}")
    resp_col = _find(df, "respondent")
    comment_col = _find(df, "comments")

    wide_cols = {int(m.group(1)): c for c in df.columns if (m := WIDE_RANK_PATTERN.match(str(c)))}
    rank_col, year_col = _find(df, "rank"), _find(df, "year")
    if rank_col and year_col:
        info["format"] = "long (one row per respondent x rank)"
        long = df.rename(columns={rank_col: "rank", year_col: "year"})
    elif wide_cols:
        info["format"] = f"wide ({', '.join(wide_cols[k] for k in sorted(wide_cols))}) -> melted to long"
        id_vars = [c for c in df.columns if c not in wide_cols.values()]
        long = df.melt(id_vars=id_vars, value_vars=list(wide_cols.values()),
                       var_name="_rank_col", value_name="year")
        long["rank"] = long["_rank_col"].map({v: k for k, v in wide_cols.items()})
        # A blank rank_N cell is a missing answer, not a record: drop it, but count it.
        n_blank = int(long["year"].isna().sum())
        if n_blank:
            info["notes"].append(f"{n_blank} blank rank cells (unanswered) were skipped, not imputed.")
        long = long[long["year"].notna()]
    else:
        raise ValueError("Could not find rank/year columns (long) or rank_1..rank_5 columns (wide). "
                         f"Columns present: {list(df.columns)}")

    if resp_col is None:
        info["notes"].append("No respondent column; each source row treated as its own respondent.")
        long["respondent"] = long["source_row"].astype(str)
    else:
        long["respondent"] = long[resp_col].astype("string")
    info["columns_used"] = {"respondent": resp_col, "location": loc_col, "rank": rank_col or "rank_1..5",
                            "year": year_col or "rank_1..5 values", "comments": comment_col}
    if resp_col == "start":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(long["respondent"], errors="coerce")
        if ts.notna().all():
            info["notes"].append("`start` parses as timestamps; used as the respondent/session key.")
        else:
            info["notes"].append("`start` holds respondent labels (not timestamps); used as respondent key.")

    long["location_raw"] = long[loc_col]
    long["comment"] = (long[comment_col] if comment_col else pd.Series(pd.NA, index=long.index))

    out = long[["source_row", "respondent", "location_raw", "rank", "year", "comment"]].copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["comment"] = out["comment"].map(_clean_comment)
    lookup = _alias_lookup(gaz)
    out["location"] = out["location_raw"].map(lambda v: lookup.get(_key(v)) if pd.notna(v) else None)

    # ---- validation: every failing row goes to `issues` with all its reasons
    reasons = pd.Series([[] for _ in range(len(out))], index=out.index)

    def flag(mask, why):
        for i in out.index[mask]:
            reasons[i].append(why)

    flag(out["rank"].isna() | (out["rank"] % 1 != 0) | ~out["rank"].isin(RANKS), "rank not an integer 1-5")
    flag(out["year"].isna() | (out["year"] % 1 != 0) | ~out["year"].between(YEAR_MIN, YEAR_MAX),
         f"year not an integer {YEAR_MIN}-{YEAR_MAX}")
    flag(out["location"].isna(), "unknown location")
    flag(out["respondent"].isna(), "missing respondent")
    flag(out.duplicated(["respondent", "rank"], keep=False) & out["rank"].notna(),
         "respondent gave the same rank twice")
    flag(out.duplicated(["respondent", "year"], keep=False) & out["year"].notna(),
         "respondent ranked the same year twice")
    n_loc = out.groupby("respondent")["location"].transform("nunique")
    flag(n_loc > 1, "respondent has more than one location")

    bad = reasons.map(bool)
    issues = out[bad].assign(reason=reasons[bad].map("; ".join))
    clean = out[~bad].copy()
    clean["rank"] = clean["rank"].astype(int)
    clean["year"] = clean["year"].astype(int)
    clean = clean.drop(columns="location_raw").sort_values(["location", "respondent", "rank"]).reset_index(drop=True)
    return clean, issues.reset_index(drop=True), info


def _clean_comment(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return pd.NA
    s = str(v).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else pd.NA


# --------------------------------------------------------------------------- statistics

def respondent_counts(clean: pd.DataFrame, gaz: pd.DataFrame) -> pd.Series:
    """Respondents per location (all 7 locations, zeros kept)."""
    return clean.groupby("location")["respondent"].nunique().reindex(gaz["survey_name"], fill_value=0)


def year_location_stats(clean: pd.DataFrame, gaz: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per gazetteer location for `year`. Locations with no data are kept (NaN ranks)."""
    n_total = respondent_counts(clean, gaz)
    sel = clean[clean["year"] == year]
    g = sel.groupby("location")["rank"]
    stats = pd.DataFrame({
        "n_selected_year": g.size(),
        "mean_rank": g.mean(),
        "median_rank": g.median(),
    })
    dist = sel.pivot_table(index="location", columns="rank", values="respondent",
                           aggfunc="count", fill_value=0).reindex(columns=RANKS, fill_value=0)
    dist.columns = [f"rank_{r}" for r in RANKS]
    out = gaz.set_index("survey_name")[["upazila", "district", "lat", "lon", "note"]].copy()
    out["n_respondents"] = n_total
    out = out.join(stats).join(dist)
    out["n_selected_year"] = out["n_selected_year"].fillna(0).astype(int)
    for c in dist.columns:
        out[c] = out[c].fillna(0).astype(int)
    out["share_selected"] = out["n_selected_year"] / out["n_respondents"].where(out["n_respondents"] > 0)
    out["status"] = "ranked"
    out.loc[out["n_selected_year"] == 0, "status"] = "not ranked this year"
    out.loc[out["n_respondents"] == 0, "status"] = "no survey data"
    out["small_sample"] = (out["n_respondents"] > 0) & (out["n_respondents"] < SMALL_SAMPLE_N)
    out.index.name = "location"
    return out.reset_index()


def year_frequency(clean: pd.DataFrame, location: str | None = None) -> pd.DataFrame:
    """Mentions of each year (2000-2020) in respondents' top five, split by rank.

    A zero means "nobody put this year in their top five", not "no flood".
    """
    d = clean if location is None else clean[clean["location"] == location]
    tab = (d.pivot_table(index="year", columns="rank", values="respondent", aggfunc="count", fill_value=0)
           .reindex(index=range(YEAR_MIN, YEAR_MAX + 1), columns=RANKS, fill_value=0))
    tab.columns = [f"rank_{r}" for r in RANKS]
    tab["mentions"] = tab.sum(axis=1)
    tab.index.name = "year"
    return tab.reset_index()


def top_years(clean: pd.DataFrame, location: str | None = None, k: int = 5) -> pd.DataFrame:
    """Most frequently reported years; ties broken by lower (worse) mean rank."""
    d = clean if location is None else clean[clean["location"] == location]
    n_resp = d["respondent"].nunique()
    if d.empty:
        return pd.DataFrame(columns=["year", "mentions", "share_of_respondents", "mean_rank", "median_rank",
                                     "times_ranked_worst"])
    g = d.groupby("year")["rank"]
    t = pd.DataFrame({"mentions": g.size(), "mean_rank": g.mean().round(2), "median_rank": g.median(),
                      "times_ranked_worst": g.apply(lambda s: int((s == 1).sum()))})
    t["share_of_respondents"] = (t["mentions"] / n_resp).round(2)
    t = t.reset_index().sort_values(["mentions", "mean_rank", "year"], ascending=[False, True, True])
    return t.head(k)[["year", "mentions", "share_of_respondents", "mean_rank", "median_rank",
                      "times_ranked_worst"]].reset_index(drop=True)


def comments(clean: pd.DataFrame, location: str | None = None) -> pd.DataFrame:
    """One row per respondent with a non-empty comment (deduplicated across their rank rows)."""
    d = clean if location is None else clean[clean["location"] == location]
    c = d.dropna(subset=["comment"]).drop_duplicates(["respondent", "comment"])
    return c[["location", "respondent", "comment"]].reset_index(drop=True)


# --------------------------------------------------------------------------- bundle

@dataclass
class SurveyData:
    path: str
    raw: pd.DataFrame
    schema: pd.DataFrame
    gazetteer: pd.DataFrame
    clean: pd.DataFrame
    issues: pd.DataFrame
    info: dict = field(default_factory=dict)

    @property
    def looks_like_test_data(self) -> bool:
        return bool(self.clean["respondent"].astype(str).str.match(r"(?i)^test\s*user").any())


def load_dataset(path: str | Path = DEFAULT_PARQUET, gazetteer_path: str | Path = GAZETTEER_CSV) -> SurveyData:
    raw = load_raw(path)
    gaz = load_gazetteer(gazetteer_path)
    clean, issues, info = normalize(raw, gaz)
    return SurveyData(str(path), raw, inspect_schema(raw), gaz, clean, issues, info)
