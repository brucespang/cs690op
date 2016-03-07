from collections import defaultdict
import csv
import numpy as np

error = defaultdict(list)

with open("baseline.csv", 'r') as f:
    reader = csv.reader(f, delimiter=" ")
    for (alg, dataset, rmse) in reader:
        error[alg].append(float(rmse))

with open("complex.csv", 'r') as f:
    reader = csv.reader(f, delimiter=" ")
    for (alg, dataset, rmse) in reader:
        error[alg].append(float(rmse))

print "Algorithm, Part 1, Part 2, Part 3, Part 4, Part 5, Average"
for alg,es in sorted(error.items(), key=lambda x: x[0]):
    print alg.replace('_', ' ').title() + ", " + ', '.join(["%0.4f"%(e) for e in es]) + ", %0.4f"%(np.mean(es))
