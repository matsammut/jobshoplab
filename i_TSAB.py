import random
import heapq
import math
from collections import defaultdict
import numpy as np
import time
from pathlib import Path

class Job:
    def __init__(self, job_id, operations):
        self.job_id = job_id
        self.operations = operations  # list of (machine_id, proc_time)

class Operation:
    def __init__(self, job, index, machine, duration):
        self.job = job
        self.index = index
        self.machine = machine
        self.duration = duration

class Schedule:
    def __init__(self, jobs):
        self.jobs = jobs
        self.operations = [
            Operation(j.job_id, idx, m, d)
            for j in jobs
            for idx, (m, d) in enumerate(j.operations)
        ]
        self.machine_sequences = defaultdict(list)

    def random_initial(self):
        for op in self.operations:
            self.machine_sequences[op.machine].append(op)
        for m in self.machine_sequences:
            random.shuffle(self.machine_sequences[m])

    def evaluate(self):
        """Compute makespan with disjunctive graph forward simulation."""
        job_end = defaultdict(int)
        machine_end = defaultdict(int)
        for machine, seq in self.machine_sequences.items():
            time = 0
            for op in seq:
                start = max(job_end[op.job], time)
                finish = start + op.duration
                job_end[op.job] = finish
                time = finish
                machine_end[machine] = finish
        return max(job_end.values())

    def critical_path(self):
        """Return machines/ops that are on the current critical path."""
        # Simplified placeholder
        return list(self.operations)

class TabuList:
    def __init__(self):
        self.tabu_dict = {}

    def is_tabu(self, move, iteration):
        return self.tabu_dict.get(move, -1) > iteration

    def add(self, move, iteration, tenure):
        self.tabu_dict[move] = iteration + tenure

# def i_tsab(jobs, max_iter=1000, tabu_tenure=10):
#     sched = Schedule(jobs)
#     sched.random_initial()
#     best_sched = sched
#     best_makespan = sched.evaluate()
#     tabu = TabuList()
#     iteration = 0
#     while iteration < max_iter:
#         iteration += 1
#         moves = []  # generate moves from critical path
#         crit_ops = sched.critical_path()
#         for i in range(len(crit_ops) - 1):
#             m = crit_ops[i].machine
#             if crit_ops[i+1].machine == m:
#                 move = (crit_ops[i], crit_ops[i+1])
#                 moves.append(move)

#         candidate = None
#         cand_value = float("inf")
#         for move in moves:
#             # swap
#             m = move[0].machine
#             seq = sched.machine_sequences[m]
#             idx1 = seq.index(move[0])
#             idx2 = seq.index(move[1])
#             seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
#             val = sched.evaluate()
#             seq[idx1], seq[idx2] = seq[idx2], seq[idx1]  # undo
#             if not tabu.is_tabu(move, iteration) or val < best_makespan:
#                 if val < cand_value:
#                     cand_value = val
#                     candidate = move

#         if candidate:
#             m = candidate[0].machine
#             seq = sched.machine_sequences[m]
#             idx1 = seq.index(candidate[0])
#             idx2 = seq.index(candidate[1])
#             seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
#             tabu.add(candidate, iteration, tabu_tenure)
#             if cand_value < best_makespan:
#                 best_makespan = cand_value
#                 best_sched = sched

#     return best_sched, best_makespan


# Example usage of i_tsab()

# Define jobs (job_id, [(machine_id, duration), ...])
# jobs = [
#     Job(0, [(0, 3), (1, 2), (2, 2)]),
#     Job(1, [(0, 2), (2, 1), (1, 4)]),
#     Job(2, [(1, 4), (2, 3), (0, 2)])
# ]

# # Run i-TSAB
# best_sched, best_makespan = i_tsab(jobs, max_iter=500, tabu_tenure=5)

# print("Best makespan found:", best_makespan)

# # Print out machine sequences
# for m, seq in best_sched.machine_sequences.items():
#     print(f"Machine {m} sequence:", [(op.job, op.index) for op in seq])

#Replicate the experiments, 10 independeent trails of each of the 50 benchmark instances and record the makespan of the best solution located during each trial.




# Assume i_tsab() and sts() are implemented
# Both return (best_schedule, best_makespan)

def load_taillard(filepath):
    """Load a Taillard benchmark instance from .txt file."""
    with open(filepath, "r") as f:
        data = list(map(int, f.read().split()))
    n_jobs, n_machines = data[:2]
    ops = data[2:]
    jobs = []
    for j in range(n_jobs):
        job_ops = []
        for k in range(n_machines):
            machine = ops[2*(j*n_machines+k)]
            duration = ops[2*(j*n_machines+k)+1]
            job_ops.append((machine, duration))
        jobs.append(Job(j, job_ops))
    return jobs

def run_experiment(instances_dir, algorithms, n_trials=10, max_iter=int(5e7)):
    results = {}
    for file in sorted(Path(instances_dir).glob("ta*.txt")):
        inst_name = file.stem
        jobs = load_taillard(file)
        results[inst_name] = {}
        for alg_name, alg_func in algorithms.items():
            makespans = []
            for trial in range(n_trials):
                start = time.time()
                _, best_makespan = alg_func(jobs, max_iter=max_iter)
                makespans.append(best_makespan)
                print(f"{inst_name} {alg_name} trial {trial+1}: {best_makespan} (time {time.time()-start:.1f}s)")
            results[inst_name][alg_name] = {
                "best": np.min(makespans),
                "avg": np.mean(makespans)
            }
    return results

def print_table(results):
    header = f"{'Instance':<8} | {'STS Best':<10} {'STS Avg':<10} | {'i-TSAB Best':<12} {'i-TSAB Avg':<12}"
    print(header)
    print("-"*len(header))
    for inst, vals in results.items():
        print(f"{inst:<8} | {vals['STS']['best']:<10} {vals['STS']['avg']:<10.1f} | {vals['i-TSAB']['best']:<12} {vals['i-TSAB']['avg']:<12.1f}")

import random
import copy

def sts(jobs, max_iter=100000, tabu_tenure=10):
    # Initial solution: dispatching rule (random order)
    schedule = random_schedule(jobs)
    best_schedule, best_makespan = schedule, makespan(schedule)
    
    tabu_list = {}
    current = schedule
    
    for it in range(max_iter):
        neighborhood = generate_neighbors(current)
        best_neighbor, best_val = None, float("inf")
        
        for move, neighbor in neighborhood:
            if move in tabu_list and tabu_list[move] > it:
                continue  # tabu
            val = makespan(neighbor)
            if val < best_val:
                best_neighbor, best_val = neighbor, val
        
        if best_neighbor is None:
            continue
        
        current = best_neighbor
        if best_val < best_makespan:
            best_schedule, best_makespan = copy.deepcopy(current), best_val
        
        # update tabu list
        move = extract_move(current)
        tabu_list[move] = it + tabu_tenure
    
    return best_schedule, best_makespan

def i_tsab(jobs, max_iter=100000, tenure_min=5, tenure_max=20, max_no_improve=1000):
    schedule = random_schedule(jobs)
    best_schedule, best_makespan = schedule, makespan(schedule)
    
    tabu_list = {}
    current = schedule
    no_improve = 0
    
    for it in range(max_iter):
        neighborhood = generate_neighbors(current)
        best_neighbor, best_val, best_move = None, float("inf"), None
        
        for move, neighbor in neighborhood:
            if move in tabu_list and tabu_list[move] > it:
                continue
            val = makespan(neighbor)
            if val < best_val:
                best_neighbor, best_val, best_move = neighbor, val, move
        
        if best_neighbor is None:
            continue
        
        current = best_neighbor
        if best_val < best_makespan:
            best_schedule, best_makespan = copy.deepcopy(current), best_val
            no_improve = 0
        else:
            no_improve += 1
        
        # adaptive tenure
        adaptive_tenure = random.randint(tenure_min, tenure_max)
        tabu_list[best_move] = it + adaptive_tenure
        
        # perturbation when stagnating
        if no_improve > max_no_improve:
            current = perturb_solution(current)
            no_improve = 0
    
    return best_schedule, best_makespan


algorithms = {
    "STS": sts,
    "i-TSAB": i_tsab
}

results = run_experiment("data/taillard", algorithms, n_trials=10, max_iter=int(5e7))
print_table(results)
