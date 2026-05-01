"""
Download all required WRDS data sources and save as parquet files under data/raw/.
Run once: python -m data.wrds_download
Takes ~30-60 minutes depending on connection speed.

Combines CRSP Legacy (v1, up to 2024-12-31) with CIZ (v2, 2025-01-01+) via UNION ALL
so the resulting parquet files are seamless across the schema migration.
"""

import wrds
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


def get_db():
    return wrds.Connection()


def save(df: pd.DataFrame, name: str):
    path = RAW_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {name}: {len(df):,} rows → {path}")


# ---------------------------------------------------------------------------
# CRSP
# ---------------------------------------------------------------------------

def download_crsp_dsf(db: wrds.Connection):
    """Daily stock file: prices, returns, volume. UNION of v1 and v2."""
    df = db.raw_sql("""
        SELECT permno, date, ABS(prc) AS prc, ret, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date >= '2000-01-01' AND date <= '2024-12-31'

        UNION ALL

        SELECT permno, date, ABS(prc) AS prc, ret, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf_v2
        WHERE date >= '2025-01-01'
    """, date_cols=["date"])
    save(df, "crsp_dsf")


def download_crsp_dsedelist(db: wrds.Connection):
    """Delisting dates, codes, and returns. UNION of v1 and v2."""
    df = db.raw_sql("""
        SELECT permno, dlstdt, dlstcd, dlret
        FROM crsp.dsedelist
        WHERE dlstdt >= '2000-01-01' AND dlstdt <= '2024-12-31'

        UNION ALL

        SELECT permno, dlstdt, dlstcd, dlret
        FROM crsp.dsedelist_v2
        WHERE dlstdt >= '2025-01-01'
    """, date_cols=["dlstdt"])
    save(df, "crsp_dsedelist")


def download_crsp_dsp500list(db: wrds.Connection):
    """Historical S&P 500 constituent membership. UNION of v1 and v2."""
    df = db.raw_sql("""
        SELECT permno, start, ending
        FROM crsp.dsp500list

        UNION ALL

        SELECT permno, start, ending
        FROM crsp.dsp500list_v2
        WHERE start > (SELECT MAX(start) FROM crsp.dsp500list)
    """, date_cols=["start", "ending"])
    df = df.drop_duplicates(subset=["permno", "start"])
    save(df, "crsp_dsp500list")


# ---------------------------------------------------------------------------
# Compustat
# ---------------------------------------------------------------------------

def download_compustat_fundq(db: wrds.Connection):
    """Quarterly fundamentals via CCM link (v1 + v2). capxy renamed to capxq."""
    df = db.raw_sql("""
        SELECT
            b.lpermno AS permno,
            a.gvkey, a.datadate, a.rdq, a.fqtr, a.fyearq,
            a.saleq, a.cogsq, a.xsgaq, a.oibdpq, a.niq, a.epspxq,
            a.atq, a.ltq, a.ceqq, a.dlttq, a.dlcq, a.cheq,
            a.capxy, a.dvy, a.cshoq, a.prccq,
            a.rectq, a.invtq, a.apq, a.actq, a.lctq,
            a.ibq, a.dpq, a.txpq, a.oancfy
        FROM comp.fundq a
        JOIN (
            SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
            FROM crsp.ccmxpf_lnkhist
            UNION ALL
            SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
            FROM crsp.ccmxpf_lnkhist_v2
            WHERE linkdt > (SELECT MAX(linkdt) FROM crsp.ccmxpf_lnkhist)
        ) b ON a.gvkey = b.gvkey
            AND b.linktype IN ('LC','LU','LS')
            AND b.linkprim IN ('P','C')
            AND a.datadate BETWEEN b.linkdt AND COALESCE(b.linkenddt, CURRENT_DATE)
        WHERE a.indfmt = 'INDL'
          AND a.datafmt = 'STD'
          AND a.popsrc = 'D'
          AND a.consol = 'C'
          AND a.datadate >= '1999-01-01'
        ORDER BY b.lpermno, a.datadate
    """, date_cols=["datadate", "rdq"])
    df = df.rename(columns={"capxy": "capxq"})
    save(df, "compustat_fundq")


# ---------------------------------------------------------------------------
# WRDS Financial Ratios
# ---------------------------------------------------------------------------

def download_wrds_ratios(db: wrds.Connection):
    """WRDS Financial Ratios Library — pre-computed ratios keyed on permno + public_date."""
    df = db.raw_sql("""
        SELECT permno, public_date,
               bm, pe_op_basic, pe_op_dil, pe_exi, pe_inc,
               ps, pcf, dpr, npm, opmad, opmbd, gpm, ptpm, cfm,
               roe, roa, capital_ratio, equity_invcap, debt_invcap,
               totdebt_invcap, efftax, aftret_eq, aftret_invcapx,
               pretret_noa, pretret_earnat, lt_debt, dltt_be, debt_at,
               debt_ebitda, curr_ratio, quick_ratio, cash_ratio,
               inv_turn, at_turn, rect_turn, pay_turn, sale_invcap,
               rd_sale, adv_sale, staff_sale, accrual,
               cash_conversion, intcov, intcov_ratio, ocf_lct,
               cash_lt, invt_act, rect_act, debt_assets, debt_capital,
               de_ratio, dltt_eq, evm, peg_trailing,
               divyield, short_ratio
        FROM wrdsapps_finratio.firm_ratio
        WHERE public_date >= '1999-01-01'
    """, date_cols=["public_date"])
    save(df, "wrds_ratios")


# ---------------------------------------------------------------------------
# I/B/E/S
# ---------------------------------------------------------------------------

def download_ibes_statsum(db: wrds.Connection):
    """I/B/E/S summary stats: analyst estimates. SUE computed as (actual-mean)/stdev."""
    df = db.raw_sql("""
        SELECT ticker, statpers, fpedats, fpi, measure,
               numest, meanest, medest, stdev, highest, lowest, actual
        FROM ibes.statsum_epsus
        WHERE statpers >= '1999-01-01'
          AND fpi IN ('0','1','2','3','4')
          AND measure = 'EPS'
    """, date_cols=["statpers", "fpedats"])
    # Compute standardised unexpected earnings here since suescore not in all subscriptions
    df["sue"] = (df["actual"] - df["meanest"]) / df["stdev"].replace(0, float("nan"))
    save(df, "ibes_statsum")


def download_ibes_ticker_permno(db: wrds.Connection):
    """IBES ticker to PERMNO mapping via CRSP link."""
    df = db.raw_sql("""
        SELECT ticker, permno, sdate, edate
        FROM wrdsapps.ibcrsphist
    """, date_cols=["sdate", "edate"])
    save(df, "ibes_ticker_permno")


# ---------------------------------------------------------------------------
# Fama-French Factors
# ---------------------------------------------------------------------------

def download_ff_factors(db: wrds.Connection):
    """Fama-French 5-factor + momentum daily returns."""
    df = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff.fivefactors_daily
        WHERE date >= '1999-01-01'
    """, date_cols=["date"])
    save(df, "ff_factors")


# ---------------------------------------------------------------------------
# Betas — computed from rolling regressions on FF factors
# ---------------------------------------------------------------------------

def download_beta_suite(db: wrds.Connection):
    """
    WRDS beta tables vary by subscription. Try known table names in order.
    Falls back to a rolling-beta stub that feature_store.py will compute at runtime.
    """
    candidates = [
        # Try WRDS beta suite tables (subscription-dependent)
        ("""
            SELECT permno, date,
                   beta AS betamkt, b_smb AS betasmb,
                   b_hml AS betahml, b_mom AS betamom
            FROM crsp.betas
            WHERE date >= '1999-01-01'
         """, ["date"]),
        ("""
            SELECT permno, date,
                   betamkt, betasmb, betahml, betamom
            FROM wrdsapps.betas
            WHERE date >= '1999-01-01'
         """, ["date"]),
    ]

    for sql, date_cols in candidates:
        try:
            df = db.raw_sql(sql, date_cols=date_cols)
            save(df, "beta_suite")
            return
        except Exception:
            continue

    # No beta table accessible — save an empty stub; feature_store will handle missing betas
    print("  Beta suite table not found in this subscription — saving empty stub.")
    stub = pd.DataFrame(columns=["permno", "date", "betamkt", "betasmb", "betahml", "betamom"])
    save(stub, "beta_suite")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Connecting to WRDS...")
    db = get_db()

    steps = [
        ("CRSP DSF (daily prices)",         download_crsp_dsf),
        ("CRSP DSEdelist (delistings)",      download_crsp_dsedelist),
        ("CRSP DSP500List (S&P 500 history)",download_crsp_dsp500list),
        ("Compustat FundQ via CCM",          download_compustat_fundq),
        ("WRDS Financial Ratios",            download_wrds_ratios),
        ("I/B/E/S Summary Stats",            download_ibes_statsum),
        ("I/B/E/S Ticker-PERMNO map",        download_ibes_ticker_permno),
        ("Fama-French 5 Factors Daily",      download_ff_factors),
        ("WRDS Beta Suite",                  download_beta_suite),
    ]

    for label, fn in steps:
        print(f"\n--- {label} ---")
        try:
            fn(db)
        except Exception as e:
            print(f"ERROR in {label}: {e}")

    db.close()
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()