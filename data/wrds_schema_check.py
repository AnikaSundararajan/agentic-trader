"""
Diagnostic script to check which WRDS v2 tables and columns exist in your subscription.
Run: python3.11 -m data.wrds_schema_check
Paste the output back so queries in wrds_download.py can be fixed.
"""

import wrds

db = wrds.Connection()

print("\n=== v2 / CIZ tables in crsp schema ===")
crsp_tables = db.list_tables(library="crsp")
v2_tables = [t for t in crsp_tables if "_v2" in t or "ciz" in t]
print(v2_tables if v2_tables else "  None found")

print("\n=== All crsp tables (for reference) ===")
print(sorted(crsp_tables))

print("\n=== Columns for v2 tables we need ===")
for library, table in [
    ("crsp", "dsf_v2"),
    ("crsp", "dsedelist_v2"),
    ("crsp", "dsp500list_v2"),
    ("crsp", "ccmxpf_lnkhist_v2"),
]:
    print(f"\n--- {library}.{table} ---")
    try:
        desc = db.describe_table(library=library, table=table)
        print(desc.to_string())
    except Exception as e:
        print(f"  Not found: {e}")

print("\n=== wrdsapps_finratio.firm_ratio columns ===")
try:
    desc = db.describe_table(library="wrdsapps_finratio", table="firm_ratio")
    print(desc.to_string())
except Exception as e:
    print(f"  Error: {e}")

print("\n=== ibes.statsum_epsus columns ===")
try:
    desc = db.describe_table(library="ibes", table="statsum_epsus")
    print(desc.to_string())
except Exception as e:
    print(f"  Error: {e}")

db.close()
print("\nDone.")