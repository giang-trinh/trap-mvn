import subprocess
import os
import time
import csv

from bench_utils import bench_trapmvn, bench_trappist, bench_mpbn

def run_benchmark(output: str, REPETITIONS, TIMEOUT, SOLUTIONS):
    with open(output, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        
        header = ["Model"]
        header += ["trapmvn (general)", "trapmvn (unitary)"]
        writer.writerow(header)
        print(header)
        
        for model in sorted(os.listdir("models")):
            if not model.endswith(".sbml"):
                continue
            
            row = [model]
            model = f"models/{model}"
            # Fixed points are shared across semantics.
            trapmvn_general = bench_trapmvn(model, semantics="general", problem="max", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            trapmvn_unitary = bench_trapmvn(model, semantics="unitary", problem="max", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(trapmvn_general), str(trapmvn_unitary)]
            print(row)
            
            writer.writerow(row)
            csvfile.flush()

run_benchmark('max-trap-benchmark.time-to-first.tsv', REPETITIONS = 1, TIMEOUT = 3600, SOLUTIONS = 1)
run_benchmark('max-trap-benchmark.time-to-all.tsv', REPETITIONS = 1, TIMEOUT = 3600, SOLUTIONS = 100_000)