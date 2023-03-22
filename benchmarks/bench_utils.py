import subprocess
import os
import time

def bench_trapmvn(model, semantics = "general", problem = "min", fix = "deadlock", REPETITIONS = 3, TIMEOUT = 3600, SOLUTIONS = 1):  
    cmd_trapmvn = ["python3", "helper_trapmvn.py", model, problem, semantics, str(SOLUTIONS), fix]
    total_time = 0
    fails = 0
    result_count = None
    for _i in range(REPETITIONS):
        try:
            start = time.perf_counter()
            result = subprocess.run(cmd_trapmvn, capture_output=True, timeout=TIMEOUT)
            total_time += time.perf_counter() - start
            new_result_count = int(result.stdout.splitlines()[0].strip())

            if result.returncode != 0:
                fails += 1
            elif result_count is not None and new_result_count != result_count:
                fails += 1
            result_count = new_result_count            
        except subprocess.TimeoutExpired:
            total_time += TIMEOUT
            fails += 1
    return (float(total_time) / float(REPETITIONS), fails, result_count)

def bench_trappist(model, problem = "min", fix = "1", REPETITIONS = 3, TIMEOUT = 3600, SOLUTIONS = 1):   
    model_bnet = model.replace(".sbml", ".bnet")
    cmd_trappist = ["python3", "helper_trappist.py", model_bnet, problem, str(SOLUTIONS), fix]
    total_time = 0
    fails = 0
    result_count = None
    for _i in range(REPETITIONS):
        try:
            start = time.perf_counter()
            # First, create a bnet model file.
            result = subprocess.call(['java', '-jar', 'bioLQM.jar', model, model_bnet], stdout=subprocess.DEVNULL)
            assert result == 0            
            result = subprocess.run(cmd_trappist, capture_output=True, timeout=TIMEOUT)
            total_time += time.perf_counter() - start
            new_result_count = int(result.stdout.splitlines()[0].strip())

            if result.returncode != 0:
                fails += 1
            elif result_count is not None and new_result_count != result_count:
                fails += 1
            result_count = new_result_count            
        except subprocess.TimeoutExpired:
            total_time += TIMEOUT
            fails += 1
    return (float(total_time) / float(REPETITIONS), fails, result_count)

def bench_mpbn(model, problem = "min", REPETITIONS = 3, TIMEOUT = 3600, SOLUTIONS = 1):
    model_bnet = model.replace(".sbml", ".bnet")
    total_time = 0
    fails = 0
    result_count = 0
    for _i in range(REPETITIONS):
        try:
            start = time.perf_counter()
            # First, convert .sbml to .bnet
            result = subprocess.call(['java', '-jar', 'bioLQM.jar', model, model_bnet], stdout=subprocess.DEVNULL)
            assert result == 0
            # Then use mpbn
            output = subprocess.run(['python3', 'helper_mpbn.py', model_bnet, problem, str(SOLUTIONS)], capture_output=True, timeout=TIMEOUT)
            total_time += time.perf_counter() - start            
            result_count = len(output.stdout.splitlines())  
            if output.returncode != 0:
                fails += 1
        except subprocess.TimeoutExpired:
            total_time += TIMEOUT
            fails += 1
    return (float(total_time) / float(REPETITIONS), fails, result_count) 