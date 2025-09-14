import os
import random
import copy
import numpy as np
from collections import defaultdict

# ==========================================================
# 1. Parser for Taillard JSP benchmark files
# ==========================================================

def parse_taillard(file_path):
    """
    Parse Taillard benchmark file into job operations.
    Returns jobs as list of lists [(machine, proc_time), ...]
    """
    with open(file_path, "r") as f:
        content = f.read().split()

    n_jobs = int(content[0])
    n_machines = int(content[1])
    numbers = list(map(int, content[2:]))

    jobs = []
    for j in range(n_jobs):
        job = []
        for m in range(n_machines):
            machine = numbers[(j * n_machines + m) * 2]
            duration = numbers[(j * n_machines + m) * 2 + 1]
            job.append((machine, duration))
        jobs.append(job)
    return jobs, n_jobs, n_machines


# ==========================================================
# 2. Schedule representation + makespan calculation
# ==========================================================

def compute_makespan(jobs, job_order):
    """
    jobs: list of job operations
    job_order: sequence of job indices, each repeated n_machines times
    """
    n_jobs = len(jobs)
    n_machines = len(jobs[0])

    # track completion times
    job_completion = [0] * n_jobs
    machine_completion = [0] * n_machines
    job_op_index = [0] * n_jobs

    for j in job_order:
        op_idx = job_op_index[j]
        machine, duration = jobs[j][op_idx]

        start_time = max(job_completion[j], machine_completion[machine])
        finish_time = start_time + duration

        job_completion[j] = finish_time
        machine_completion[machine] = finish_time
        job_op_index[j] += 1

    return max(job_completion)


def random_schedule(jobs):
    """
    Generate random valid schedule (job permutation with repeated jobs).
    """
    n_jobs = len(jobs)
    n_machines = len(jobs[0])
    order = []
    for j in range(n_jobs):
        order += [j] * n_machines
    random.shuffle(order)
    return order


# ==========================================================
# 3. Neighborhood: swap adjacent ops of different jobs
# ==========================================================

def generate_neighbors(schedule):
    neighbors = []
    for i in range(len(schedule) - 1):
        if schedule[i] != schedule[i + 1]:
            new_sched = schedule[:]
            new_sched[i], new_sched[i + 1] = new_sched[i + 1], new_sched[i]
            move = (schedule[i], schedule[i + 1], i)
            neighbors.append((move, new_sched))
    return neighbors


def perturb_solution(schedule, strength=5):
    sched = schedule[:]
    for _ in range(strength):
        i, j = random.sample(range(len(sched)), 2)
        sched[i], sched[j] = sched[j], sched[i]
    return sched


# ==========================================================
# 4. STS and i-TSAB implementations
# ==========================================================

def sts(jobs, max_iter=10000, tabu_tenure=10):
    current = random_schedule(jobs)
    best = current
    best_val = compute_makespan(jobs, current)

    tabu_list = {}
    for it in range(max_iter):
        neighborhood = generate_neighbors(current)
        best_neighbor, best_val_n, best_move = None, float("inf"), None
        for move, sched in neighborhood:
            if move in tabu_list and tabu_list[move] > it:
                continue
            val = compute_makespan(jobs, sched)
            if val < best_val_n:
                best_neighbor, best_val_n, best_move = sched, val, move

        if best_neighbor is None:
            continue

        current = best_neighbor
        if best_val_n < best_val:
            best, best_val = current, best_val_n

        tabu_list[best_move] = it + tabu_tenure

    return best, best_val


def i_tsab(jobs, max_iter=10000, tenure_min=5, tenure_max=20, max_no_improve=200):
    current = random_schedule(jobs)
    best = current
    best_val = compute_makespan(jobs, current)

    tabu_list = {}
    no_improve = 0

    for it in range(max_iter):
        neighborhood = generate_neighbors(current)
        best_neighbor, best_val_n, best_move = None, float("inf"), None
        for move, sched in neighborhood:
            if move in tabu_list and tabu_list[move] > it:
                continue
            val = compute_makespan(jobs, sched)
            if val < best_val_n:
                best_neighbor, best_val_n, best_move = sched, val, move

        if best_neighbor is None:
            continue

        current = best_neighbor
        if best_val_n < best_val:
            best, best_val = current, best_val_n
            no_improve = 0
        else:
            no_improve += 1

        adaptive_tenure = random.randint(tenure_min, tenure_max)
        tabu_list[best_move] = it + adaptive_tenure

        if no_improve > max_no_improve:
            current = perturb_solution(current)
            no_improve = 0

    return best, best_val


# ==========================================================
# 5. Experiment runner
# ==========================================================

def run_experiment(instances_path, algorithms, trials=10, max_iter=20000):
    results = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(instances_path)):
        if not fname.startswith("ta"):
            continue
        if int(fname[-2:]) > 10:
            continue
        instance = os.path.join(instances_path, fname)
        jobs, _, _ = parse_taillard(instance)

        for alg_name, alg in algorithms.items():
            for t in range(trials):
                _, val = alg(jobs, max_iter=max_iter)
                results[fname][alg_name].append(val)
                print(f"{fname} {alg_name} trial {t+1}: {val}")
    return results


def summarize_results(results):
    print("\n=== Summary Results ===")
    for inst, algs in results.items():
        print(f"\nInstance {inst}")
        for alg, vals in algs.items():
            print(f"{alg}: best={min(vals)} avg={np.mean(vals):.2f} std={np.std(vals):.2f}")



algorithms = {
    "STS": sts,
    "i-TSAB": i_tsab
}

results = run_experiment(
    instances_path="data/jssp_instances",
    algorithms=algorithms,
    trials=3,
    max_iter=500  # increase to ~5e7 for full-scale experiments
)

summarize_results(results)
