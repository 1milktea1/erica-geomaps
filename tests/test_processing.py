"""Tests for processing.py and the Streamlit app.

The bad-row fixtures below are built in memory to exercise validation; they are
never written to disk or shown in the app.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import processing as P  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return P.load_dataset()


def test_parquet_loads(data):
    assert data.raw.shape == (35, 5)
    assert list(data.raw.columns) == ["start", "loc_question", "rank", "year", "comments"]
    assert data.info["columns_used"]["location"] == "loc_question"


def test_normalize_all_valid(data):
    assert len(data.clean) == 35 and data.issues.empty
    assert data.clean["year"].between(2000, 2020).all()
    assert set(data.clean["rank"]) == {1, 2, 3, 4, 5}
    assert data.clean["comment"].isna().all()  # no comments in this file; none invented
    assert data.looks_like_test_data


def test_every_survey_location_mapped(data):
    assert set(data.clean["location"]) == set(data.gazetteer["survey_name"])
    assert len(data.gazetteer) == 7


def test_stats_2004_match_raw(data):
    s = P.year_location_stats(data.clean, data.gazetteer, 2004).set_index("location")
    expected = {"Chilmari": 1, "Fulcchari": 2, "Kurigram Sadar": 3, "Sadullapur": 1,
                "Sariakandi": 3, "Sherpur": 2, "Sughatta": 4}
    assert s["mean_rank"].to_dict() == {k: float(v) for k, v in expected.items()}
    assert s["n_selected_year"].sum() == 7
    raw = data.raw[data.raw.year == 2004]
    assert raw["rank"].mean() == pytest.approx(16 / 7)
    assert s["small_sample"].all()


def test_unranked_year_is_not_zero_severity(data):
    s = P.year_location_stats(data.clean, data.gazetteer, 2010)
    assert (s["n_selected_year"] == 0).all()
    assert s["mean_rank"].isna().all()  # not 0, not imputed
    assert (s["status"] == "not ranked this year").all()


def test_year_frequency_totals(data):
    f = P.year_frequency(data.clean)
    assert len(f) == 21 and f["mentions"].sum() == 35
    assert f.set_index("year").loc[2004, ["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]].tolist() == [2, 2, 2, 1, 0]
    loc = P.year_frequency(data.clean, "Chilmari")
    assert loc["mentions"].sum() == 5


def test_top_years_chilmari(data):
    t = P.top_years(data.clean, "Chilmari")
    assert t["year"].tolist() == [2004, 2020, 2017, 2019, 2007]  # all mentioned once; ordered by rank


def test_validation_rejects_bad_rows(data):
    bad = pd.DataFrame({
        "start": ["A", "A", "B", "C", "C", "D"],
        "loc_question": ["Chilmari", "Chilmari", "Atlantis", "Sherpur", "Sherpur", " fulchhari "],
        "rank": [1, 6, 1, 2, 2, 1],
        "year": [1999, 2004, 2004, 2007, 2017, 2020],
        "comments": [None, None, None, None, None, "  "],
    })
    clean, issues, _ = P.normalize(bad, data.gazetteer)
    reasons = issues.set_index("source_row")["reason"]
    assert "year not an integer" in reasons[0]
    assert "rank not an integer" in reasons[1]
    assert "unknown location" in reasons[2]
    assert "same rank twice" in reasons[3] and "same rank twice" in reasons[4]
    assert clean["location"].tolist() == ["Fulcchari"]  # alias + whitespace/case resolved
    assert clean["comment"].isna().all()  # blank comment stays missing


def test_wide_format_is_melted(data):
    wide = pd.DataFrame({"start": ["X"], "loc_question": ["Sughatta"], "rank_1": [2004],
                         "rank_2": [2007], "rank_3": [None], "rank_4": [2019], "rank_5": [2020],
                         "comments": ["river rose fast"]})
    clean, issues, info = P.normalize(wide, data.gazetteer)
    assert issues.empty and len(clean) == 4  # blank rank_3 skipped, not imputed
    assert any("blank rank cells" in n for n in info["notes"])
    assert P.comments(clean)["comment"].tolist() == ["river rose fast"]


def test_coordinates_inside_own_upazila(data):
    geo = json.loads(P.BOUNDARIES_GEOJSON.read_text())
    polys = {f["properties"]["survey_name"]: shape(f["geometry"]) for f in geo["features"]}
    for _, r in data.gazetteer.iterrows():
        pt = Point(r["lon"], r["lat"])
        assert polys[r["survey_name"]].contains(pt), r["survey_name"]
        others = [k for k, p in polys.items() if k != r["survey_name"] and p.contains(pt)]
        assert not others
    g = data.gazetteer.set_index("survey_name")
    assert g.loc["Sherpur", "district"] == "Bogura"
    assert g.loc["Sughatta", "upazila"] == "Saghata"


# --------------------------------------------------------------------------- app

def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_year_slider_updates():
    at = _app()
    assert _metric(at, "Respondents who ranked 2004") == "7 of 7"
    at.slider[0].set_value(2010).run()
    assert not at.exception
    assert _metric(at, "Respondents who ranked 2010") == "0 of 7"
    assert _metric(at, "Overall mean rank for 2010") == "—"
    at.radio[0].set_value("Median rank").run()
    at.slider[0].set_value(2019).run()
    assert _metric(at, "Overall median rank for 2019") == "4.00"


def test_app_location_filter():
    at = _app()
    at.selectbox[0].set_value("Sherpur").run()
    assert not at.exception
    assert _metric(at, "Respondents") == "1"
    assert _metric(at, "Ranked answers") == "5"
    assert any("Sherpur Upazila, Bogura" in h.value for h in at.subheader)
    assert any("INFERRED" in c.value for c in at.caption)
    assert any("No written comments" in c.value for c in at.caption)
