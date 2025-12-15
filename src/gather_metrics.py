#!/usr/bin/env python3
import pandas as pd
import glob
import sys
import os


def main():
    # Set paths
    input_dir = "output/metrics/metabolites/"
    output_file = "output/metrics/consolidated_metabolites_metrics.csv"

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(csv_files)} CSV files")

    # Read and concatenate all files
    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"✓ {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Error reading {os.path.basename(file_path)}: {e}")

    if not dataframes:
        print("❌ No valid CSV files could be read")
        sys.exit(1)

    # Concatenate and save
    consolidated_df = pd.concat(dataframes, ignore_index=True)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save consolidated file
    consolidated_df.to_csv(output_file, index=False)

    print(f"\n✅ SUCCESS!")
    print(f"📁 Files aggregated: {len(dataframes)}")
    print(f"📊 Total rows: {len(consolidated_df)}")
    print(f"📋 Columns: {', '.join(consolidated_df.columns)}")
    print(f"💾 Output: {output_file}")


if __name__ == "__main__":
    main()
