#!/usr/bin/env python3
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import polars as pl
import requests


def fetch_and_process_dataset(ekey, output_filename):
    # Hardcoded path prefix for parquet files
    path = 'https://s3.kausal.tech/datasets/'

    try:
        # Fetch the XML content
        response = requests.get(path)  # noqa: S113
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the XML
        root = ET.fromstring(response.content)  # noqa: S314

        # First, search for direct key matches
        found = False
        for content in root.findall('.//{*}Contents'):
            key_element = content.find('.//{*}Key')
            if key_element is not None:
                key = key_element.text
                if ekey == key:
                    found = True
                    print(f"Found direct key match: {key}")
                    
                    try:
                        print(f"Reading parquet file from: {path + key}")
                        df = pl.read_parquet(path + key)

                        print(f"Saving dataframe to: {output_filename}")
                        df.write_csv(output_filename)
                        print(f"Successfully saved dataframe to {output_filename}")
                        return True  # noqa: TRY300
                    except Exception as e:
                        print(f"Error processing parquet file: {e}")
                        return False

        # If no direct key match, search for matching ETag in Contents elements
        if not found:
            for content in root.findall('.//{*}Contents'):
                etag_element = content.find('.//{*}ETag')
                if etag_element is not None:
                    etag_value = etag_element.text.strip('"')

                    if ekey in etag_value:
                        found = True
                        key_element = content.find('.//{*}Key')
                        if key_element is not None:
                            key = key_element.text
                            print(f"Found matching dataset with Key: {key}")
                    else:
                        key = 'files/md5/' + ekey[:2] + '/' + ekey[2:]
                        print(f"Didn't find key but trying with {key}.")

                    try:
                        print(f"Reading parquet file from: {path + key}")
                        df = pl.read_parquet(path + key)

                        print(f"Saving dataframe to: {output_filename}")
                        df.write_csv(output_filename)
                        print(f"Successfully saved dataframe to {output_filename}")
                        return True  # noqa: TRY300
                    except Exception as e:
                        print(f"Error processing parquet file: {e}")
                        return False

        if not found:
            print(f"No datasets found with ETag containing '{ekey}'")
        return found  # noqa: TRY300

    except requests.exceptions.RequestException as e:
        print(f"Error fetching dataset: {e}")
        return False
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fetch dataset by ekey and save as CSV.')
    parser.add_argument('ekey', help='The ekey to search for in the dataset.')
    parser.add_argument('output', help='The filename for the output CSV file.')
    args = parser.parse_args()

    fetch_and_process_dataset(args.ekey, args.output)

if __name__ == "__main__":
    main()
