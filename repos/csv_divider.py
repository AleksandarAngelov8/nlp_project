import csv
import os
import argparse

def split_csv(input_file, output_dir, chunk_size=50000):
    """
    Splits a CSV file into smaller chunks with a fixed number of lines.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory where output chunks will be saved.
        chunk_size (int): Number of lines per chunk (default = 50,000).
    """
    input_file = os.path.abspath(input_file)   # ensure absolute path
    output_dir = os.path.abspath(output_dir)   # ensure absolute path

    os.makedirs(output_dir, exist_ok=True)     # create output folder if missing

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # capture header

        file_count = 1
        rows = []
        
        for i, row in enumerate(reader, start=1):
            rows.append(row)
            if i % chunk_size == 0:
                output_file = os.path.join(output_dir, f"part{file_count}.csv")
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)  # write header
                    writer.writerows(rows)
                print(f"Created {output_file}")
                file_count += 1
                rows = []

        # write remaining rows
        if rows:
            output_file = os.path.join(output_dir, f"part{file_count}.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Created {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into smaller chunks.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_dir", help="Directory where output files will be stored")
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Number of lines per chunk (default: 50000)")
    args = parser.parse_args()

    split_csv(args.input_file, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
