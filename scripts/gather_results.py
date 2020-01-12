#!/usr/bin/env python3

import sys
import os
import csv


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gather_results.py [directory with results] [output file]")
        exit(1)

    results_dir = sys.argv[1]
    output = sys.argv[2]

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
                    value = ":".join(row[1:]).strip()

                    if measure not in results_fields:
                        results_fields.append(measure)

                    result[measure] = value

        results.append(result)

    # Save results to the provided output file
    with open(output, 'w', encoding="utf-8") as file_out:
        csv_output = csv.DictWriter(file_out, fieldnames=parameters_fields + results_fields)
        csv_output.writeheader()
        for r in results:
            csv_output.writerow(r)

    print("Wrote {} rows to {}.".format(len(results), output))