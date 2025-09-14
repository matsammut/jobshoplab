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



# algorithms = {
#     "STS": sts,
#     "i-TSAB": i_tsab
# }

# results = run_experiment(
#     instances_path="data/jssp_instances",
#     algorithms=algorithms,
#     trials=3,
#     max_iter=500  # increase to ~5e7 for full-scale experiments
# )

# summarize_results(results)


import numpy as np
import pandas as pd

# 1. Best known makespans for Taillard ta01–50
# (source: Taillard 1993, also available at http://jobshop.jjvh.nl/)
BEST_KNOWN = {
    "ta01": 1231, "ta02": 1244, "ta03": 1218, "ta04": 1175, "ta05": 1224,
    "ta06": 1238, "ta07": 1227, "ta08": 1217, "ta09": 1274, "ta10": 1241,
    "ta11": 1357, "ta12": 1367, "ta13": 1342, "ta14": 1345, "ta15": 1339,
    "ta16": 1360, "ta17": 1462, "ta18": 1396, "ta19": 1332, "ta20": 1348,
    "ta21": 1642, "ta22": 1600, "ta23": 1557, "ta24": 1644, "ta25": 1595,
    "ta26": 1645, "ta27": 1680, "ta28": 1603, "ta29": 1625, "ta30": 1584,
    "ta31": 1764, "ta32": 1784, "ta33": 1791, "ta34": 1828, "ta35": 2007,
    "ta36": 1819, "ta37": 1771, "ta38": 1673, "ta39": 1795, "ta40": 1670,
    "ta41": 2006, "ta42": 1939, "ta43": 1846, "ta44": 1979, "ta45": 2000,
    "ta46": 2006, "ta47": 1889, "ta48": 1937, "ta49": 1960, "ta50": 1923,
}

def compute_re(makespan, best_known):
    return 100.0 * (makespan - best_known) / best_known

def experiment_with_mre(instances_path, algorithms, trials=10, max_iter=50000):
    results = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(instances_path)):
        if not fname.startswith("ta"):
            continue
        if int(fname[-2:]) > 50:
            continue
        if fname not in BEST_KNOWN:
            continue

        jobs, _, _ = parse_taillard(os.path.join(instances_path, fname))
        best_known = BEST_KNOWN[fname]

        for alg_name, alg in algorithms.items():
            vals = []
            for t in range(trials):
                _, val = alg(jobs, max_iter=max_iter)
                vals.append(val)
            # store RE not absolute makespan
            results[fname][alg_name] = [compute_re(v, best_known) for v in vals]
            print(f"{fname} {alg_name} trial {t+1}: {val}")

    return results

def summarize_mre(results):
    # group by instance ranges like ta01–10, ta11–20, ...
    summary = []
    groups = [(1,10),(11,20),(21,30),(31,40),(41,50)]
    #groups = [(1,10),(11,20)]
    for lo, hi in groups:
        group_name = f"ta{lo:02d}–{hi:02d}"

        row = {"Problem": group_name}
        for alg in ["STS", "i-TSAB"]:
            group_res = []
            for i in range(lo, hi+1):
                print(lo)
                inst = f"ta{i:02d}"
                print(inst)
                if inst in results:
                    group_res.extend(results[inst][alg])
            bMRE = min(group_res)
            mMRE = np.mean(group_res)
            row[f"{alg}_bMRE"] = bMRE
            row[f"{alg}_mMRE"] = mMRE
        summary.append(row)
    return pd.DataFrame(summary)


algorithms = {
    "STS": sts,
    "i-TSAB": i_tsab
}

results = experiment_with_mre(
    instances_path="data/jssp_instances",
    algorithms=algorithms,
    trials=10,
    max_iter=50000000  # for testing; scale up later
)

summarize_results(results)
df = summarize_mre(results)
print(df.to_string(index=False))
