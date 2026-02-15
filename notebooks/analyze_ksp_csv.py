#!/usr/bin/env python3
from __future__ import annotations

import sys

import polars as pl

"""Analyze CSV file contents."""

if len(sys.argv) < 2:
    print("Usage: python analyze_csv.py <file.csv>")
    sys.exit(1)

df = pl.read_csv(sys.argv[1])

print("=== Basic Statistics ===")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\n=== Columns ===")
for col in df.columns:
    print(f"  - {col}")

print("\n=== Column Info ===")
print(df.schema)

print("\n=== Unique Values ===")
for col in ['year', 'unit', 'quality', 'energy_carrier_group', 'origin']:
    if col in df.columns:
        n_unique = df[col].n_unique()
        print(f"{col}: {n_unique} unique values")
        if n_unique <= 20:
            print(f"  Values: {sorted(df[col].unique().to_list())}")
        else:
            print(f"  Sample: {sorted(df[col].unique().to_list())[:10]}...")

print("\n=== Value Statistics ===")
if 'value' in df.columns:
    print("Value column:")
    print(f"  Min: {df['value'].min()}")
    print(f"  Max: {df['value'].max()}")
    print(f"  Mean: {df['value'].mean():.2f}")
    print(f"  Non-zero values: {(df['value'] != 0).sum():,} ({(df['value'] != 0).sum() / len(df) * 100:.1f}%)")

print("\n=== Sample Rows (first 10) ===")
print(df.head(10))

print("\n=== Sample Rows with non-zero values ===")
non_zero = df.filter(pl.col('value') != 0)
if len(non_zero) > 0:
    print(non_zero.head(10))
else:
    print("No non-zero values found")
