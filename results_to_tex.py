#!/usr/bin/env python

import os

def time2sec(time):
    m, s = time.strip("s").split("m")
    s = float(m) * 60 + float(s)
    return s

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

                    #tex_line += "& {} & {}".format(time[0], est[0])
                    #tex_line += "& {}".format(time[0].strip("0m"))
                    tex_line += "& {:.1f}".format(time2sec(time[0]))
                    tex_line += " & {:.2f} & {:.1f}".format(float(recall[0]) * 100, time2sec(time[1]))
                    for i in range(1, len(u)):
                        if i > 3:
                            break
                        # if i == 1:
                        #     continue
                        #tex_line += " & {} & {} & {} & {} & {}".format(u[i], recall[i], time[i + 1], est[i + 1], pred[i])
                        #tex_line += " & {:.4f} & {:.4f} & {:.1f}".format(float(u[i]), float(recall[i]), time2sec(time[i + 1]))
                        #tex_line += " & {:.2f} & {:.2f} & {}".format(float(u[i]) * 100, float(recall[i]) * 100, time[i + 1].strip("0m").strip("s"))
                        try:
                            tex_line += " & {:.2f} & {:.1f} & {:.1f}".format(float(u[i]) * 100, float(pred[i]), time2sec(time[i + 1]))
                        except:
                            continue
                    tex_line += " \\\\\n"

                    fo.write(tex_line)