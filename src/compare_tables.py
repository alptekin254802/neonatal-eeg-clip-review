"""Compare both regenerated tables with the values printed in the manuscript."""

import argparse
import csv
from pathlib import Path


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    return [[cell.strip().replace("−", "-") for cell in row] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    reported = Path(__file__).resolve().parents[1] / "reported"
    for number in (1, 2):
        name = f"table_{number}.csv"
        expected = read_table(reported / name)
        actual = read_table(args.results / name)
        if [x.casefold().removeprefix("participant ") for x in expected[0]] != [x.casefold().removeprefix("participant ") for x in actual[0]]:
            raise SystemExit(f"{name}: column labels differ.")
        if len(actual) != len(expected):
            raise SystemExit(f"{name}: expected {len(expected) - 1} rows, found {len(actual) - 1}.")
        for left, right in zip(expected[1:], actual[1:]):
            if left != right:
                raise SystemExit(f"{name}: expected {left}, found {right}.")
        print(f"Table {number}: all {len(expected) - 1} rows match the manuscript at the reported precision.")


if __name__ == "__main__":
    main()
