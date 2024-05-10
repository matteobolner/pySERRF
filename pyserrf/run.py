#!/usr/bin/env python3
from serrf import read_and_parse_excel, SERRF
import sys

if __name__ == "__main__":
    test = read_and_parse_excel("test_data/SERRF example dataset.xlsx")

    a = SERRF(threads=4, batch_column="batch", random_state=int(sys.argv[1]))
    normalized = a.fit_transform(test)
    normalized.to_csv(sys.argv[2], sep="\t")
