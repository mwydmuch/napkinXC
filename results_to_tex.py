#!/usr/bin/env python

import os

if __name__ == "__main__":
    with open("results.tex", "w") as fo:
        for r, d, f in os.walk("results"):
            for file in sorted(f):
                print("Gathering results from {}/{} ...".format(r, file))
                tex_line = ""

                recall = []
                u = []
                time = []
                est = []
                pred = []

                with open(r + "/" + file) as fi:
                    for i, line in enumerate(fi):
                        if i == 0:
                            tex_line += "{}".format(line)
                        elif "Recall:" in line:
                            recall.append(line.split(":")[1].strip())
                        elif "user" in line:
                            time.append(line.split("\t")[1].strip())
                        elif "Mean pred. size:" in line:
                            pred.append(line.split(":")[1].strip())
                        elif "Mean # estimators per data point:" in line:
                            est.append(line.split(":")[1].strip())
                        elif "uP:" in line or "uAlfaBeta(" in line:
                            u.append(line.split(":")[1].strip())

                    tex_line += "& {} & {}".format(time[0], est[0])
                    for i in range(len(u)):
                        if i > 4:
                            break
                        if i == 1:
                            continue
                        tex_line += " & {} & {} & {} & {} & {}".format(u[i], recall[i], time[i + 1], est[i + 1], pred[i])

                    tex_line += " \\\\\n"

                    fo.write(tex_line)