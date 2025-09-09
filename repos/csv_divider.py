import csv
import os
import re
import argparse
import sys

# Safely set max field size limit
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)  # maximum for a signed 32-bit C long on Windows

PART_RE = re.compile(r"^part(\d+)\.csv$", re.IGNORECASE)

def find_last_part_number(output_dir: str) -> int:
    last = 0
    if not os.path.isdir(output_dir):
        return 0
    for name in os.listdir(output_dir):
        m = PART_RE.match(name)
        if m:
            try:
                n = int(m.group(1))
                if n > last:
                    last = n
            except ValueError:
                pass
    return last

def split_csv(input_file, output_dir, chunk_size=50000, strict=False):
    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    last_part = find_last_part_number(output_dir)
    if strict and last_part > 0:
        missing = [i for i in range(1, last_part + 1)
                   if not os.path.exists(os.path.join(output_dir, f"part{i}.csv"))]
        if missing:
            raise RuntimeError(
                f"Strict mode: missing expected parts in {output_dir}: {missing}"
            )

    rows_to_skip = last_part * chunk_size

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        if rows_to_skip:
            skipped = 0
            for _ in range(rows_to_skip):
                try:
                    next(reader)
                    skipped += 1
                except StopIteration:
                    break
            print(f"Resuming: detected part{last_part}.csv; skipped {skipped} data rows.")
            if skipped < rows_to_skip:
                print("No more rows to process. Nothing to do.")
                return

        file_count = last_part + 1
        rows = []

        for i, row in enumerate(reader, start=1):
            rows.append(row)
            if i % chunk_size == 0:
                output_file = os.path.join(output_dir, f"part{file_count}.csv")
                if os.path.exists(output_file):
                    raise FileExistsError(f"{output_file} already exists; refusing to overwrite.")
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(rows)
                print(f"Created {output_file}")
                file_count += 1
                rows = []

        if rows:
            output_file = os.path.join(output_dir, f"part{file_count}.csv")
            if os.path.exists(output_file):
                raise FileExistsError(f"{output_file} already exists; refusing to overwrite.")
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Created {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV into fixed-size chunks with auto-resume.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_dir", help="Directory where output files will be stored")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Number of data rows per chunk (default: 50000)")
    parser.add_argument("--strict", action="store_true",
                        help="Abort if parts 1..N are not all present when resuming")
    args = parser.parse_args()

    split_csv(args.input_file, args.output_dir, args.chunk_size, strict=args.strict)

if __name__ == "__main__":
    main()
