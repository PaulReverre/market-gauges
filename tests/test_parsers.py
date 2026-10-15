"""Parser tests against small hand-built fixtures (no network)."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from conftest import xlsx
from gauges.sources import (aaii, cboe, census, cftc, edgar, eia, factset, finnhub, finra, fred, gdelt, gpu_index, mops,
                            naaim, openrouter, shiller, ssga, treasury, tsmc, wikimedia, yf)
from gauges.sources.base import business_days, completed_session, each


def test_cboe_index_csv_uses_last_column():
    vix = cboe.parse_index_csv("DATE,OPEN,HIGH,LOW,CLOSE\n10/13/2026,17,18,16,17.10\n10/14/2026,15.17,17.45,14.98,16.87\n", "VIX")
    vvix = cboe.parse_index_csv("DATE,VVIX\n10/14/2026,94.45\n", "VVIX")
    assert [(o.obs_date, o.value) for o in vix] == [(date(2026, 10, 13), 17.10), (date(2026, 10, 14), 16.87)]
    assert vvix[0].value == 94.45 and vvix[0].vintage is None


def test_cboe_csv_rejects_unexpected_header():
    with pytest.raises(Exception, match="unexpected CSV header"):
        cboe.parse_index_csv("<html>", "VIX")


def test_cboe_putcall_archive_and_json():
    archive = cboe.parse_putcall_archive(
        "CBOE TOTAL PUT/CALL RATIO\nDATE,CALLS,PUTS,TOTAL,P/C Ratio\n10/04/2019, 2175006, 2289715, 4464721, 1.05\n", "totalpc")
    assert [(o.series, o.obs_date, o.value) for o in archive] == [("pc_total", date(2019, 10, 4), 1.05)]
    payload = {"ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": "0.88"},
                          {"name": "EQUITY PUT/CALL RATIO", "value": "0.61"}, {"name": "MXEF PUT/CALL RATIO", "value": "3.4"}]}
    rows = cboe.parse_putcall_json(payload, date(2026, 10, 14))
    assert {o.series: o.value for o in rows} == {"pc_total": 0.88, "pc_equity": 0.61}


def test_fred_observations_skip_missing_and_set_vintage():
    rows = [{"date": "2026-08-01", "value": "130.455", "realtime_start": "2026-09-26"},
            {"date": "2026-09-01", "value": ".", "realtime_start": "2026-10-10"}]
    plain = fred.parse_observations(rows, "fred", "PCEPILFE", with_vintage=False)
    vintaged = fred.parse_observations(rows, "alfred", "PCEPILFE", with_vintage=True)
    assert len(plain) == 1 and plain[0].vintage is None
    assert vintaged[0].vintage == date(2026, 9, 26)


def test_cftc_vintage_is_friday_release():
    rows = [{"cftc_contract_market_code": "13874A", "report_date_as_yyyy_mm_dd": "2026-10-06T00:00:00.000",
             "open_interest_all": "1932227", "asset_mgr_positions_long": "1106717"},
            {"cftc_contract_market_code": "99999X", "report_date_as_yyyy_mm_dd": "2026-10-06T00:00:00.000"}]
    out = cftc.parse_rows(rows, {"13874A": "es"}, ["open_interest_all", "asset_mgr_positions_long"])
    assert {o.series for o in out} == {"es.open_interest_all", "es.asset_mgr_positions_long"}
    assert all(o.vintage == date(2026, 10, 9) for o in out)


def test_aaii_infers_year_across_new_year():
    html = ("<table><tr><th>Reported Date</th><th>Bullish</th><th>Neutral</th><th>Bearish</th></tr>"
            "<tr><td>Jan 7</td><td>38.7%</td><td>19.7%</td><td>41.6%</td></tr>"
            "<tr><td>Dec 31</td><td>40.3%</td><td>20.8%</td><td>39.0%</td></tr></table>")
    rows = aaii.parse_aaii(html, today=date(2027, 1, 8))
    assert {(o.obs_date, o.series): o.value for o in rows}[(date(2026, 12, 31), "bearish")] == 39.0
    assert min(o.obs_date for o in rows) == date(2026, 12, 31) and len(rows) == 6


def test_btos_maps_periods_to_reference_and_publication_dates():
    content = xlsx({
        "Response Estimates": [
            ["Question ID", "Question", "Answer ID", "Answer", "202620", "202619"],
            [7, "In the last two weeks, did this business use Artificial Intelligence (AI)?", 1, "Yes", "23.8%", "."],
            [7, "In the last two weeks, did this business use Artificial Intelligence (AI)?", 2, "No", "66.4%", "."],
            [3, "Revenue question", 1, "Yes", "50%", "51%"]],
        "Collection and Reference Dates": [
            ["Sample Year", "Cycle", "Panel", "Smpdt", "Col Start", "Col End", "Ref Start", "Ref End", "Publication"],
            [None, 1, 1, 202620, datetime(2026, 9, 21), datetime(2026, 10, 4), datetime(2026, 9, 7),
             datetime(2026, 9, 20), datetime(2026, 10, 8)]],
    })
    rows = census.parse_btos(content)
    assert rows == [rows[0]] and rows[0].series == "ai_q7_yes" and rows[0].value == 23.8
    assert rows[0].obs_date == date(2026, 9, 20) and rows[0].vintage == date(2026, 10, 8)


def test_naaim_and_finra_spreadsheets():
    n = naaim.parse_naaim(xlsx({"USE_Data": [
        ["Date", "Mean/Average", "Most Bearish Response", "Quart 1", "Quart 2 (median)", "Q3", "Bull", "Standard Deviation"],
        [datetime(2026, 8, 19), 94.49, 0, 86.5, 100, 100, 200, 43.94]]}), "u")
    assert {o.series: (o.value, o.vintage) for o in n} == {
        "mean": (94.49, date(2026, 8, 20)), "median": (100, date(2026, 8, 20)), "stdev": (43.94, date(2026, 8, 20))}
    f = finra.parse_finra(xlsx({"s": [["Year-Month", "Debit", "Cash", "Margin"], ["2026-08", 1453832, 207641, 217499]]}), "u")
    assert {o.series: o.value for o in f}["debit_margin"] == 1453832 and f[0].obs_date == date(2026, 8, 1)


def test_ssga_holdings_skip_cash_and_cvr_rows():
    content = xlsx({"holdings": [
        ["Fund Name:", "SPDR S&P 500"], ["Ticker Symbol:", "SPY"], ["Holdings:", "As of 14-Oct-2026"], [],
        ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector", "Shares Held", "Local Currency"],
        ["NVIDIA CORP", "NVDA", "67066G104", "2379504", 8.05, "-", 298491438, "USD"],
        ["BERKSHIRE HATHAWAY INC CL B", "BRK.B", "084670702", "2073390", 1.6, "-", 1.0, "USD"],
        ["US DOLLAR", "-", "999USDZ92", "-", 0.28, "-", None, "USD"],
        ["TPG INC", "2602335D", "436CVR021", "-", 0.000003, "-", 1.0, "USD"], [], ["footer text"]]})
    as_of, df = ssga.parse_holdings(content)
    assert as_of == date(2026, 10, 14) and df["ticker"].tolist() == ["NVDA", "BRK.B"]
    assert ssga.yahoo_symbol("BRK.B") == "BRK-B" and ssga.yahoo_symbol("000660.KS") == "000660.KS"


def test_factset_text_tolerates_pdf_spacing_artifacts():
    text = ("Key Metrics • Earnings Growth: For Q3 2026, the estimated (year-over-year) earnings growth rate for the "
            "S&P 500 is 2 9.6%. • Earnings Guidance: For Q3 2026, 44 S&P 500 companies have issued negative EPS "
            "guidance and 72 S&P 500 companies have issued positive EPS guidance. • Valuation: The forward 12 -month "
            "P/E ratio for the S&P 500 is 19.1. This P/E ratio is below the 5-year average (19.8). Of these companies, "
            "84% have reported actual EPS above the mean EPS estimate.")
    assert factset.parse_text(text) == {"fwd_pe": 19.1, "q_growth_est": 29.6, "eps_beat_pct": 84.0,
                                        "neg_guidance": 44.0, "pos_guidance": 72.0}


@pytest.mark.parametrize("raw, expected", [(2026.01, (2026, 1)), (2026.1, (2026, 10)), (2026.12, (2026, 12)),
                                           (1871.09, (1871, 9))])
def test_shiller_decimal_months(raw, expected):
    assert shiller.parse_date(raw) == expected


def test_tsmc_and_mops_tables():
    html = ("<table><tr><th>Month</th><th>Consolidated</th></tr><tr><td>Net Revenue</td><td>YoY Change</td></tr>"
            "<tr><td>Sep.</td><td>511,857</td><td>54.6%</td></tr></table>")
    t = tsmc.parse_year(html, 2026)
    assert [(o.obs_date, o.value) for o in t] == [(date(2026, 9, 1), 511857.0)]
    page = ("<tr><td>2330</td><td>台積電</td><td>511,857,399</td><td>1</td></tr>"
            "<tr><td>1101</td><td>台泥</td><td>9,000</td></tr>")
    m = mops.parse_table(page, {"2330": "TSMC"}, 2026, 9, "u")
    assert [(o.series, o.value) for o in m] == [("2330.revenue_ntd_k", 511857399.0)]
    assert mops.months_between(date(2026, 11, 1), date(2027, 2, 1)) == [(2026, 11), (2026, 12), (2027, 1), (2027, 2)]


def test_gpu_index_csvs():
    uvik = gpu_index.parse_uvik(
        "provider,provider_type,product,gpu,gpus_per_instance,usd_per_instance_hour,usd_per_gpu_hour,pricing,captured\n"
        "AWS,hyperscaler,p5.48xlarge,H100 SXM,8,55.04,6.88,on-demand list price,2026-10-07\n", "u")
    assert uvik[0].series == "H100 SXM|AWS|p5.48xlarge|on-demand list price" and uvik[0].value == 6.88
    gph = gpu_index.parse_gpuperhour(
        "snapshot_date,gpu_slug,gpu_name,basket_version,min_price_per_gpu,p25_price_per_gpu,median_price_per_gpu,"
        "p75_price_per_gpu,max_price_per_gpu\n2026-10-14,h100-nvl,H100 NVL,2026-09,2.79,,3.11,,4.5\n"
        "2026-10-15,h100-nvl,H100 NVL,2026-09,9,,9,,9\n", before=date(2026, 10, 15))
    assert {o.series: o.value for o in gph} == {"h100-nvl.min": 2.79, "h100-nvl.median": 3.11, "h100-nvl.max": 4.5}


def test_gdelt_averages_per_day_and_drops_today():
    payload = {"timeline": [{"data": [{"date": "20261014T060000Z", "value": -2.0}, {"date": "20261014T120000Z", "value": 1.0},
                                      {"date": "20261015T100000Z", "value": 3.0}]}]}
    rows = gdelt.parse_timeline(payload, "ai", before=date(2026, 10, 15))
    assert [(o.obs_date, o.value) for o in rows] == [(date(2026, 10, 14), -0.5)]


def test_openrouter_rankings_and_models():
    rows = openrouter.parse_rankings({"data": [{"date": "2026-10-13", "model_permaslug": "deepseek/x", "total_tokens": "5046621695407"}]})
    assert rows[0].series == "tokens:deepseek/x" and rows[0].value == 5046621695407.0
    models = openrouter.parse_models({"data": [{"id": "meta/muse", "created": 1792076662, "hugging_face_id": "meta/Muse",
                                                "pricing": {"prompt": "0.00000125", "completion": "bad"}}]}, date(2026, 10, 15))
    assert models.loc[0, "author"] == "meta" and models.loc[0, "prompt_price"] == 1.25e-6
    assert models.loc[0, "completion_price"] is None


def test_edgar_facts_and_8k_items():
    payload = {"cik": 789019, "facts": {"us-gaap": {"PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
        {"start": "2025-07-01", "end": "2026-06-30", "val": 115.9e9, "form": "10-K", "fy": 2026, "fp": "FY",
         "filed": "2026-07-29", "accn": "0000950170-26-000001"},
        {"start": "2025-07-01", "end": "2026-06-30", "val": 1.0, "form": "8-K", "filed": "2026-07-29", "accn": "x"}]}}}}}
    facts = edgar.parse_companyfacts(payload, "MSFT", ["PaymentsToAcquirePropertyPlantAndEquipment"])
    assert len(facts) == 1 and facts.loc[0, "filed"] == date(2026, 7, 29)
    sub = {"filings": {"recent": {"form": ["8-K", "8-K", "10-Q"], "items": ["2.02,9.01", "5.02", ""],
                                  "filingDate": ["2026-07-29", "2026-07-01", "2026-07-29"],
                                  "acceptanceDateTime": ["2026-07-29T16:05:12.000Z", "x", "y"],
                                  "accessionNumber": ["a1", "a2", "a3"]}}}
    events = edgar.parse_8k_earnings(sub, "MSFT")
    assert events["accn"].tolist() == ["a1"] and events.loc[0, "report_date"] == date(2026, 7, 29)


def test_finnhub_calendar_and_surprises():
    cal = finnhub.parse_calendar({"earningsCalendar": [
        {"symbol": "NVDA", "date": "2026-11-17", "hour": "amc", "year": 2027, "quarter": 3, "epsEstimate": 2.5253},
        {"symbol": "ZZZZ", "date": "2026-11-17"}]}, {"NVDA"})
    assert cal["fiscal_period"].tolist() == ["2027Q3"]
    s = finnhub.parse_surprises([{"period": "2026-09-30", "actual": 2.22, "estimate": 2.1384}], "NVDA")
    assert {o.series: o.value for o in s} == {"NVDA.eps_actual": 2.22, "NVDA.eps_estimate": 2.1384}


def test_treasury_eia_wikimedia():
    t = treasury.parse_csv('Date,"3 Mo","2 Yr","10 Yr"\n10/14/2026,4.28,4.82,5.19\n', "u")
    assert {o.series: o.value for o in t} == {"3m": 4.28, "2y": 4.82, "10y": 5.19}
    e = eia.parse_rows([{"period": "2026-07", "stateid": "VA", "sales": "1234.5"}, {"period": "2026-07", "stateid": "TX", "sales": None}])
    assert [(o.series, o.obs_date) for o in e] == [("sales_com.VA", date(2026, 7, 1))]
    w = wikimedia.parse_pageviews({"items": [{"timestamp": "2026101400", "views": 5297}]}, "Nvidia", "u")
    assert w[0].series == "pageviews.Nvidia" and w[0].obs_date == date(2026, 10, 14)


def test_yfinance_frame_drops_partial_session_and_zero_actions():
    idx = pd.to_datetime(["2026-10-14", "2026-10-15"])
    cols = pd.MultiIndex.from_product([["Close", "Volume", "Dividends", "Stock Splits"], ["SPY", "^GSPC"]],
                                      names=["Price", "Ticker"])
    data = np.array([[769.6, 7718.94, 5.5e7, 0, 0, 0, 0, 0],
                     [770.2, 7728.13, 2.0e7, 0, 1.8, 0, 0, 0]])
    rows = yf.frame_to_obs(pd.DataFrame(data, index=idx, columns=cols), cutoff=date(2026, 10, 14))
    assert sorted(rows["series"]) == ["SPY.close", "SPY.volume", "^GSPC.close"]
    assert rows["obs_date"].max() == pd.Timestamp("2026-10-14")


def test_session_and_business_days():
    from gauges.sources.base import ET
    assert completed_session(datetime(2026, 10, 15, 15, 0, tzinfo=ET)) == date(2026, 10, 14)
    assert completed_session(datetime(2026, 10, 15, 17, 0, tzinfo=ET)) == date(2026, 10, 15)
    assert completed_session(datetime(2026, 10, 19, 8, 0, tzinfo=ET)) == date(2026, 10, 16)   # Monday morning
    assert business_days(date(2026, 10, 16), date(2026, 10, 19)) == [date(2026, 10, 16), date(2026, 10, 19)]


def test_each_isolates_failures():
    notes: list[str] = []
    out = each([1, 2], lambda i: [i] if i == 1 else 1 / 0, notes)
    assert out == [1] and notes[0].startswith("FAILED 2: ZeroDivisionError")
    with pytest.raises(RuntimeError, match="all 1 items failed"):
        each([2], lambda i: 1 / 0, [])


def test_alfred_diff_keeps_only_new_or_changed_values():
    from gauges.sources import alfred
    previous = {date(2026, 7, 1): 130.1, date(2026, 8, 1): 130.4}
    current = {date(2026, 7, 1): 130.1, date(2026, 8, 1): 130.455, date(2026, 9, 1): 130.9}
    rows = alfred.diff_vintages("PCEPILFE", previous, current, date(2026, 10, 30))
    assert [(o.obs_date, o.value) for o in rows] == [(date(2026, 8, 1), 130.455), (date(2026, 9, 1), 130.9)]
    assert all(o.vintage == date(2026, 10, 30) for o in rows)
