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
        header += ["trappist", "mpbn"]
        writer.writerow(header)
        print(header)
        
        for model in sorted(os.listdir("models")):
            if not model.endswith(".sbml"):
                continue
            
            row = [model]
            model = f"models/{model}"
            # Fixed points are shared across semantics.
            trapmvn_general = bench_trapmvn(model, semantics="general", problem="min", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            trapmvn_unitary = bench_trapmvn(model, semantics="unitary", problem="min", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(trapmvn_general), str(trapmvn_unitary)]
            trappist = bench_trappist(model, problem="min", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            mpbn = bench_mpbn(model, problem="min", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(trappist), str(mpbn)]
            print(row)
            
            writer.writerow(row)
            csvfile.flush()

run_benchmark('min-trap-benchmark.time-to-first.tsv', REPETITIONS = 1, TIMEOUT = 3600, SOLUTIONS = 1)
run_benchmark('min-trap-benchmark.time-to-all.tsv', REPETITIONS = 1, TIMEOUT = 3600, SOLUTIONS = 10_000_000)