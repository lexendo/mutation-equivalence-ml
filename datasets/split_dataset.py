from rdflib import Graph, Namespace, RDF
import random
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_FILE = "full_dataset.ttl"
TRAIN_FILE = "mutantbench_train.ttl"
TEST_FILE = "mutantbench_test.ttl"

TEST_RATIO = 0.2
RANDOM_SEED = 42

# =========================
# NAMESPACES
# =========================
MB = Namespace("https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/")

# =========================
# LOAD FULL DATASET
# =========================
print("Loading dataset...")
full_graph = Graph()
full_graph.parse(INPUT_FILE, format="turtle")

# =========================
# FIND ALL MUTANTS
# =========================
print("Extracting mutants...")
mutants = set(full_graph.subjects(RDF.type, MB.Mutant))
mutants = set(mutants)

# =========================
# GROUP MUTANTS BY PROGRAM
# =========================
print("Grouping mutants by program...")
program_to_mutants = defaultdict(list)

for mutant in mutants:
    program = full_graph.value(mutant, MB.program)
    if program is not None:
        program_to_mutants[program].append(mutant)

programs = list(program_to_mutants.keys())

print(f"Found {len(mutants)} mutants in {len(programs)} programs")

# =========================
# BALANCED RANDOM PROGRAM SPLIT (large + small guaranteed)
# =========================
print("\nPerforming balanced random program-level split...")

total_mutants = sum(len(v) for v in program_to_mutants.values())
target_test_mutants = int(total_mutants * 0.25)


programs_sorted = sorted(
    programs,
    key=lambda p: len(program_to_mutants[p]),
    reverse=True
)

random.seed(RANDOM_SEED)

# define size buckets
large_cutoff = max(1, int(0.05 * len(programs_sorted)))
small_cutoff = max(3, int(0.30 * len(programs_sorted)))

large_programs = programs_sorted[:large_cutoff]
middle_programs = programs_sorted[large_cutoff:-small_cutoff]
small_programs = programs_sorted[-small_cutoff:]

test_programs = set()
test_mutant_count = 0

chosen_large = random.choice(large_programs)
test_programs.add(chosen_large)
test_mutant_count += len(program_to_mutants[chosen_large])

chosen_smalls = random.sample(
    small_programs,
    k=min(3, len(small_programs))
)

for prog in chosen_smalls:
    test_programs.add(prog)
    test_mutant_count += len(program_to_mutants[prog])

remaining_programs = [
    p for p in programs
    if p not in test_programs
]

random.shuffle(remaining_programs)

for program in remaining_programs:
    if test_mutant_count >= target_test_mutants:
        break
    test_programs.add(program)
    test_mutant_count += len(program_to_mutants[program])

train_programs = set(programs) - test_programs

# =========================
# REPORT
# =========================
print("\n=== PROGRAM SPLIT SUMMARY ===")

print("\nTEST PROGRAMS:")
for program in sorted(test_programs, key=lambda p: len(program_to_mutants[p]), reverse=True):
    print(f"  - {program.split('#')[-1]} ({len(program_to_mutants[program])} mutants)")

print("\nTRAIN PROGRAMS:")
for program in sorted(train_programs, key=lambda p: len(program_to_mutants[p]), reverse=True):
    print(f"  - {program.split('#')[-1]} ({len(program_to_mutants[program])} mutants)")

print("\n=== MUTANT COUNTS ===")
print(f"Test mutants:  {sum(len(program_to_mutants[p]) for p in test_programs)}")
print(f"Train mutants: {sum(len(program_to_mutants[p]) for p in train_programs)}")
print(f"Total mutants: {total_mutants}")


# =========================
# CREATE TRAIN / TEST GRAPHS
# =========================
train_graph = Graph()
test_graph = Graph()

# Copy namespaces
for prefix, namespace in full_graph.namespaces():
    train_graph.bind(prefix, namespace)
    test_graph.bind(prefix, namespace)

# =========================
# COPY ALL NON-MUTANT TRIPLES
# =========================
print("Copying shared (non-mutant) triples...")

for s, p, o in full_graph:
    if s in mutants:
        continue
    train_graph.add((s, p, o))
    test_graph.add((s, p, o))

# =========================
# ADD MUTANTS TO TRAIN / TEST
# =========================
print("Adding mutants to respective splits...")

for program, mutant_list in program_to_mutants.items():
    target_graph = train_graph if program in train_programs else test_graph

    for mutant in mutant_list:
        for p, o in full_graph.predicate_objects(mutant):
            target_graph.add((mutant, p, o))

# =========================
# SAVE OUTPUT
# =========================
print("Writing output files...")
train_graph.serialize(TRAIN_FILE, format="turtle")
test_graph.serialize(TEST_FILE, format="turtle")

print("Done.")
print(f"Train dataset: {TRAIN_FILE}")
print(f"Test dataset:  {TEST_FILE}")
