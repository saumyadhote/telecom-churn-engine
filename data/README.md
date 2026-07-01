# Data

## Source

This project uses the **Company A Telecom Dataset** provided as part of the
GCI World 2026 Spring Final Assignment by Matsuo-Iwasawa Laboratory,
The University of Tokyo.

The raw CSVs are **not included** in this repository (proprietary dataset).

## Files expected

Place the following files in this directory before running any code:

| File | Rows | Columns | Description |
|---|---|---|---|
| `Client.csv` | 100,000 | 50 | Customer demographics, device info, geography, credit class |
| `Record.csv` | 100,000 | 51 | Usage behavior, billing, call quality, churn label |

They are joined on `Customer_ID`.

## Sample input for the API

`sample_input.json` contains a single synthetic customer record you can use
to test the `/predict` endpoint without the full dataset.
