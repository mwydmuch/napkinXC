#!/usr/bin/env python3

import sys
import os
import csv
from statistics import mean, stdev


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gather_results.py [directory with results] [output file] [accumulate]")
        exit(1)

    results_dir = sys.argv[1]
    output = sys.argv[2]
    accumulate = False
    if len(sys.argv) > 3 and sys.argv[3] in ['1', 'True', 'true']:
        accumulate = True

    parameters_fields = ["dataset"]  # dataset is a first parameter
    results_fields = []
    results = []

    for file in os.listdir(results_dir):  # Iterate over all files in the provided directory
        result = {}  # Parameters and measure values for a current file

        # Read parameters from the filename
        parameters = file.split("_")
        dataset = ""
        i = 0
        while i < len(parameters):
            p = parameters[i]
            if "-" == p[:1]:  # If it starts with "-" then it's program parameter
                p_name = p.strip("-")
                if p_name not in parameters_fields:
                    parameters_fields.append(p_name)
                result[p_name] = parameters[i + 1]  # The parameter name is followed by it's value
                i += 2
            else:  # Otherwise it's the name of a dataset / experiment
                if len(dataset):
                    dataset += "_"
                dataset += p
                i += 1

        result["dataset"] = dataset

        # Read results from the file
        with open(results_dir + "/" + file, "r") as file_in:
            for row in file_in:
                if ":" in row and ":" != row.strip()[-1]:
                    row = row.strip().split(":")
                    measure = row[0]
                    try:
                        value = float(row[1])
                    except ValueError:
                        continue

                    if measure not in results_fields:
                        results_fields.append(measure)

                    result[measure] = value

        results.append(result)

    # Accumulate results for different seeds and save to the provided output file
    if accumulate:
        agg_results = {}
        for r in results:
            # Build key
            key_parameters = [(k, v) for k, v in r.items() if k in parameters_fields and k != "seed"]
            run_parameters = [(k, v) for k, v in r.items() if k not in parameters_fields]
            key = "_".join(sorted(["{}:{}".format(p[0], p[1]) for p in key_parameters]))

            if key not in agg_results:
                agg_results[key] = {}
                for p in key_parameters:
                    agg_results[key][p[0]] = p[1]
                agg_results[key]['count'] = 0

            agg_results[key]['count'] += 1
            for p in run_parameters:
                agg_results[key][p[0]] = agg_results[key].get(p[0], []) + [p[1]]

        accumulated_results = []
        for k, r in agg_results.items():
            result = {}
            for rk, rv in r.items():
                if type(rv) is list:
                    result[rk + " (mean)"] = mean(rv)
                    result[rk + " (std)"] = stdev(rv)
                else:
                    result[rk] = rv
            accumulated_results.append(result)

        fields = parameters_fields + ["count"]
        for r in results_fields:
            fields.extend([r + " (mean)", r + " (std)"])
        results = accumulated_results
    else:
        fields = parameters_fields + results_fields

    with open(output, 'w', encoding="utf-8") as file_out:
        csv_output = csv.DictWriter(file_out, fieldnames=fields)
        csv_output.writeheader()
        for r in results:
            csv_output.writerow(r)

    print("Wrote {} rows to {}.".format(len(results), output))