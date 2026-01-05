from rdflib import Graph, Namespace, RDF
import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_FILE = "datasets/mutantbench_train.ttl"
OUTPUT_FILE = "mutantbench_train.csv"

# =========================
# NAMESPACES
# =========================
MB = Namespace("https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/")

# =========================
# LOAD RDF DATASET
# =========================
print("Loading RDF dataset...")
graph = Graph()
graph.parse(INPUT_FILE, format="turtle")

# =========================
# HELPER: PARSE DIFF
# =========================
def parse_diff(diff_text: str):
    before_lines = []
    after_lines = []

    for line in diff_text.splitlines():
        line = line.strip()

        if not line or line.startswith("@@"):
            continue

        if line.startswith("-"):
            before_lines.append(line[1:].strip())
        elif line.startswith("+"):
            after_lines.append(line[1:].strip())

    before = " ".join(before_lines)
    after = " ".join(after_lines)

    return before, after

# =========================
# EXTRACT MUTANTS
# =========================
print("Extracting mutants into DataFrame...")

rows = []

for mutant in graph.subjects(RDF.type, MB.Mutant):
    mutant_id = mutant.split("#")[-1]

    program = graph.value(mutant, MB.program)
    program_name = program.split("#")[-1] if program else None

    operator = graph.value(mutant, MB.operator)
    operator_name = operator.split("#")[-1] if operator else None

    diff = graph.value(mutant, MB.difference)
    diff_text = str(diff) if diff else ""

    before, after = parse_diff(diff_text)

    label = graph.value(mutant, MB.equivalence)
    label_value = 1 if str(label).lower() == "true" else 0

    rows.append({
        "mutant_id": mutant_id,
        "program": program_name,
        "operator": operator_name,
        "before": before,
        "after": after,
        "label": label_value
    })

df = pd.DataFrame(rows)

# =========================
# BASIC SANITY CHECKS
# =========================
print("\nDataFrame created.")
print(df.head())
print("\nLabel distribution:")
print(df["label"].value_counts())

# =========================
# SAVE DATAFRAME
# =========================
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDataFrame saved to {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
