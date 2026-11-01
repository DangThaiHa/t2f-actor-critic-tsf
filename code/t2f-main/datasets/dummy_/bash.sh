#!/bin/bash

# Bash script to run the Python script with the `ds_prep` function.

# Define values for variables
DIR=""
FNAME="dummy_"
FFORM=".csv"
TS_INDEX="timestamp"
L_LABEL="index"
VARIATE_COLS="['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
       '22', '23', '24', '25', '26', '27', '28', '29', 'class',
       'bc_adj_log_optimizer']"
CONT_FACTORS="['bc_adj_log_optimizer']"
TARGET_COL="bc_target_sqrt"
TIMESTAMP_FORMAT="%Y-%m-%d"
N_JOBS=2

# Function to display usage
usage() {
    echo "Usage: $0 [-d DIR] [-f FNAME] [-o FFORM] [-t TS_INDEX] [-l L_LABEL] [-v VARIATE_COLS] [-c CONT_FACTORS] [-a TARGET_COL] [-s TIMESTAMP_FORMAT] [-n N_JOBS]"
    exit 1
}

# Parse arguments
while getopts ":d:f:o:t:l:v:c:a:s:n:" opt; do
    case $opt in
        d) DIR="$OPTARG" ;;
        f) FNAME="$OPTARG" ;;
        o) FFORM="$OPTARG" ;;
        t) TS_INDEX="$OPTARG" ;;
        l) L_LABEL="$OPTARG" ;;
        v) VARIATE_COLS="$OPTARG" ;;
        c) CONT_FACTORS="$OPTARG" ;;
        a) TARGET_COL="$OPTARG" ;;
        s) TIMESTAMP_FORMAT="$OPTARG" ;;
        n) N_JOBS="$OPTARG" ;;
        *) usage ;;
    esac
done

# Run the Python script
python3 - <<EOF
import sys
import os
sys.path.insert(0, '..')
from helpers import ds_prep
import tqdm

# Ensure proper handling of progress bar in non-interactive environments
if not sys.stdout.isatty():
    tqdm.tqdm.write = lambda x: print(x)

ds_prep(
    dir="${DIR}",
    fname="${FNAME}",
    fform="${FFORM}",
    ts_index="${TS_INDEX}",
    l_label="${L_LABEL}",
    variate_cols=${VARIATE_COLS},
    cont_factors=${CONT_FACTORS},
    target_col="${TARGET_COL}",
    timestamp_format="${TIMESTAMP_FORMAT}",
    n_jobs=${N_JOBS}
)
EOF