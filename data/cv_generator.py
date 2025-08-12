import random
from atom.training.config_options import TG80MoleculeType

BASE_SEED = 42
molecules = [m.value for m in TG80MoleculeType]


def write_toml_split(filename: str, train: list[str], val: list[str], test: list[str]) -> None:
    with open(filename, "w") as f:
        _ = f.write("train_molecules = [\n")
        for m in train:
            _ = f.write(f'    "{m}",\n')
        _ = f.write("]\n\n")

        _ = f.write("validation_molecules = [\n")
        for m in val:
            _ = f.write(f'    "{m}",\n')
        _ = f.write("]\n\n")

        _ = f.write("test_molecules = [\n")
        for m in test:
            _ = f.write(f'    "{m}",\n')
        _ = f.write("]\n")


# Shuffle molecules once with a fixed seed
random.seed(BASE_SEED)
shuffled = molecules.copy()
random.shuffle(shuffled)

# Calculate fold size and number of folds
n_molecules = len(shuffled)
n_folds = 5
fold_size = n_molecules // n_folds

for fold in range(n_folds):
    # Calculate indices for this fold
    test_start = fold * fold_size
    test_end = test_start + fold_size
    val_start = test_end
    val_end = val_start + fold_size

    # Handle the case where we need to wrap around for validation
    if val_end > n_molecules:
        val_end = n_molecules
        val_start = 0

    # Get the splits
    test = shuffled[test_start:test_end]
    val = shuffled[val_start:val_end]

    # Get training set by excluding test and validation
    train = [m for m in shuffled if m not in test and m not in val]

    filename = f"data/crossval_folds/fold{fold + 1}.toml"
    write_toml_split(filename, train, val, test)
    print(f"Wrote {filename}: train={len(train)}, val={len(val)}, test={len(test)}")
