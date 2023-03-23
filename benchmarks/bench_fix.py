import subprocess
import os
import time
import csv

from bench_utils import bench_trapmvn, bench_trappist, bench_mpbn, bench_an_asp

def run_benchmark(output: str, REPETITIONS, TIMEOUT, SOLUTIONS):
    with open(output, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        
        header = ["Model"]
        header += ["trapmvn (siphon)", "trapmvn (deadlock)"]
        header += ["trappist (siphon)", "trappist (deadlock)"]
        header += ["mpbn"]
        header += ["an-asp"]
        writer.writerow(header)
        print(header)
        
        for model in sorted(os.listdir("models")):
            if not model.endswith(".sbml"):
                continue
            
            row = [model]
            model = f"models/{model}"
            # Fixed points are shared across semantics.
            trapmvn_siphon = bench_trapmvn(model, semantics="unitary", problem="fix", fix="siphon", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            trapmvn_deadlock = bench_trapmvn(model, semantics="unitary", problem="fix", fix="deadlock", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)        
            row += [str(trapmvn_siphon), str(trapmvn_deadlock)]
            trappist_1 = bench_trappist(model, problem="fix", fix="1", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            trappist_2 = bench_trappist(model, problem="fix", fix="2", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(trappist_1), str(trappist_2)]
            mpbn = bench_mpbn(model, problem="fix", REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(mpbn)]
            an_asp = bench_an_asp(model, REPETITIONS=REPETITIONS, TIMEOUT=TIMEOUT, SOLUTIONS=SOLUTIONS)
            row += [str(an_asp)]
            print(row)
            
            writer.writerow(row)
            csvfile.flush()

run_benchmark('fix-benchmark.time-to-first.tsv', REPETITIONS = 1, TIMEOUT = 3600, SOLUTIONS = 1)