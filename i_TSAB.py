import random
import heapq
import math
from collections import defaultdict

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

def i_tsab(jobs, max_iter=1000, tabu_tenure=10):
    sched = Schedule(jobs)
    sched.random_initial()
    best_sched = sched
    best_makespan = sched.evaluate()
    tabu = TabuList()
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        moves = []  # generate moves from critical path
        crit_ops = sched.critical_path()
        for i in range(len(crit_ops) - 1):
            m = crit_ops[i].machine
            if crit_ops[i+1].machine == m:
                move = (crit_ops[i], crit_ops[i+1])
                moves.append(move)

        candidate = None
        cand_value = float("inf")
        for move in moves:
            # swap
            m = move[0].machine
            seq = sched.machine_sequences[m]
            idx1 = seq.index(move[0])
            idx2 = seq.index(move[1])
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
            val = sched.evaluate()
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]  # undo
            if not tabu.is_tabu(move, iteration) or val < best_makespan:
                if val < cand_value:
                    cand_value = val
                    candidate = move

        if candidate:
            m = candidate[0].machine
            seq = sched.machine_sequences[m]
            idx1 = seq.index(candidate[0])
            idx2 = seq.index(candidate[1])
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
            tabu.add(candidate, iteration, tabu_tenure)
            if cand_value < best_makespan:
                best_makespan = cand_value
                best_sched = sched

    return best_sched, best_makespan


# Example usage of i_tsab()

# Define jobs (job_id, [(machine_id, duration), ...])
jobs = [
    Job(0, [(0, 3), (1, 2), (2, 2)]),
    Job(1, [(0, 2), (2, 1), (1, 4)]),
    Job(2, [(1, 4), (2, 3), (0, 2)])
]

# Run i-TSAB
best_sched, best_makespan = i_tsab(jobs, max_iter=500, tabu_tenure=5)

print("Best makespan found:", best_makespan)

# Print out machine sequences
for m, seq in best_sched.machine_sequences.items():
    print(f"Machine {m} sequence:", [(op.job, op.index) for op in seq])

#Replicate the experiments, 10 independeent trails of each of the 50 benchmark instances and record the makespan of the best solution located during each trial.

