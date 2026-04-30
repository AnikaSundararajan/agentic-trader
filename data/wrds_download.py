"""
Download all required WRDS data sources and save as parquet files under data/raw/.
Run once: python -m data.wrds_download
Takes ~30-60 minutes depending on connection speed.
"""

import os
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


def download_crsp_dsf(db: wrds.Connection):
    """Daily stock file: prices, returns, volume. 2000-01-01 onward."""
    df = db.raw_sql("""
        SELECT permno, date, prc, ret, vol, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date >= '2000-01-01'
    """, date_cols=["date"])
    save(df, "crsp_dsf")


def download_crsp_dsedelist(db: wrds.Connection):
    """Delisting dates, codes, and returns."""
    df = db.raw_sql("""
        SELECT permno, dlstdt, dlstcd, dlret
        FROM crsp.dsedelist
        WHERE dlstdt >= '2000-01-01'
    """, date_cols=["dlstdt"])
    save(df, "crsp_dsedelist")


def download_crsp_dsp500list(db: wrds.Connection):
    """Historical S&P 500 constituent membership (point-in-time)."""
    df = db.raw_sql("""
        SELECT permno, start, ending
        FROM crsp.dsp500list
    """, date_cols=["start", "ending"])
    save(df, "crsp_dsp500list")


def download_compustat_fundq(db: wrds.Connection):
    """Quarterly fundamentals via CCM link. Includes rdq (report date) for lag."""
    df = db.raw_sql("""
        SELECT
            b.lpermno AS permno,
            a.gvkey, a.datadate, a.rdq, a.fqtr, a.fyearq,
            a.saleq, a.cogsq, a.xsgaq, a.oibdpq, a.niq, a.epspxq,
            a.atq, a.ltq, a.ceqq, a.dlttq, a.dlcq, a.cheq,
            a.capxq, a.dvy, a.cshoq, a.prccq,
            a.rectq, a.invtq, a.apq, a.actq, a.lctq,
            a.ibq, a.dpq, a.txpq, a.oancfy
        FROM comp.fundq a
        JOIN crsp.ccmxpf_lnkhist b
            ON a.gvkey = b.gvkey
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
    save(df, "compustat_fundq")


def download_wrds_ratios(db: wrds.Connection):
    """WRDS Financial Ratios Library — 70+ pre-computed ratios keyed on permno + public_date."""
    df = db.raw_sql("""
        SELECT permno, public_date,
               bm, pe_op_basic, pe_op_dil, pe_exi, pe_inc,
               ps, pcf, dpr, npm, opmad, opmbd, gpm, ptpm, cfm,
               roe, roa, capital_ratio, equity_invcap, debt_invcap,
               totdebt_invcap, efftax, aftret_eq, aftret_invcapx,
               pretret_noa, pretret_earnat, GrLTDebt, dltt_be, debt_at,
               debt_ebitda, curr_ratio, quick_ratio, cash_ratio,
               inv_turn, at_turn, rect_turn, pay_turn, sale_invcap,
               rd_sale, adv_sale, staff_sale, accrual,
               cash_conversion, IntCov, intcov_ratio, ocf_lct,
               cash_lt, inv_act, rect_act, debt_assets, debt_capital,
               de_ratio, intcov, dltt_eq, evm, PEG_trailing,
               divyield, short_ratio
        FROM wrdsapps_finratio.firm_ratio
        WHERE public_date >= '1999-01-01'
    """, date_cols=["public_date"])
    save(df, "wrds_ratios")


def download_ibes_statsum(db: wrds.Connection):
    """I/B/E/S summary stats: analyst estimates, SUE, revisions."""
    df = db.raw_sql("""
        SELECT ticker, statpers, fpedats, fpi, measure,
               numest, meanest, medest, stdev, highest, lowest,
               actual, surpmean, surpmed, suescore
        FROM ibes.statsum_epsus
        WHERE statpers >= '1999-01-01'
          AND fpi IN ('0','1','2','3','4')
          AND measure = 'EPS'
    """, date_cols=["statpers", "fpedats"])
    save(df, "ibes_statsum")


def download_ibes_ticker_permno(db: wrds.Connection):
    """IBES ticker to PERMNO mapping via CRSP link."""
    df = db.raw_sql("""
        SELECT ticker, permno, sdate, edate
        FROM wrdsapps.ibcrsphist
    """, date_cols=["sdate", "edate"])
    save(df, "ibes_ticker_permno")


def download_ff_factors(db: wrds.Connection):
    """Fama-French 5-factor daily returns."""
    df = db.raw_sql("""
        SELECT date, mktrf, smb, hml, rmw, cma, rf, umd
        FROM ff.fivefactors_daily
        WHERE date >= '1999-01-01'
    """, date_cols=["date"])
    save(df, "ff_factors")


def download_beta_suite(db: wrds.Connection):
    """WRDS Beta Suite: stock-level factor betas."""
    df = db.raw_sql("""
        SELECT permno, date, beta, betamkt, betasmb, betahml, betarmw, betacma, betamom
        FROM wrdsapps_beta.beta
        WHERE date >= '1999-01-01'
    """, date_cols=["date"])
    save(df, "beta_suite")


def main():
    print("Connecting to WRDS...")
    db = get_db()

    steps = [
        ("CRSP DSF (daily prices)", download_crsp_dsf),
        ("CRSP DSEdelist (delistings)", download_crsp_dsedelist),
        ("CRSP DSP500List (S&P 500 history)", download_crsp_dsp500list),
        ("Compustat FundQ via CCM", download_compustat_fundq),
        ("WRDS Financial Ratios", download_wrds_ratios),
        ("I/B/E/S Summary Stats", download_ibes_statsum),
        ("I/B/E/S Ticker-PERMNO map", download_ibes_ticker_permno),
        ("Fama-French 5 Factors Daily", download_ff_factors),
        ("WRDS Beta Suite", download_beta_suite),
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
