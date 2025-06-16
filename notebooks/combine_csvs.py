import sys
import glob
import csv
from pathlib import Path

def combine_csv_files(input_patterns, output_file):
    """Combine multiple CSV files into one, handling different column structures automatically"""

    input_files = []
    for pattern in input_patterns:
        files = glob.glob(pattern)
        input_files.extend([Path(f) for f in files])

    if not input_files:
        print("No input files found!")
        return

    print(f"Found {len(input_files)} files to combine")

    all_columns = []
    file_headers = {}

    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                file_header = next(reader)
                file_headers[input_file] = file_header

                for col in file_header:
                    if col not in all_columns:
                        all_columns.append(col)

        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            continue

    if not all_columns:
        print("No valid CSV files found!")
        return

    total_rows = 0
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(all_columns)

        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as infile:
                    reader = csv.DictReader(infile, delimiter=';')

                    file_rows = 0
                    for row in reader:
                        output_row = []
                        for col in all_columns:
                            value = row.get(col, '')
                            if value in ['""', '"', '\'\'', "'"]:
                                value = ''
                            output_row.append(value)

                        writer.writerow(output_row)
                        file_rows += 1

                    total_rows += file_rows

            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue

    with open(output_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    content = content.replace('""', '')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

    print(f"Combined {total_rows} rows into {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_csvs.py <input_pattern1> [input_pattern2] ... <output_file>")
        sys.exit(1)

    input_patterns = sys.argv[1:-1]
    output_file = sys.argv[-1]

    try:
        combine_csv_files(input_patterns, output_file)
    except Exception as e:
        print(f"Error during combination: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()