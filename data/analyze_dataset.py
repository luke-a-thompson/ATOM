from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import gzip
import tempfile
import os
from enum import StrEnum

RDLogger.DisableLog("rdApp.*")


class MD17Smiles(StrEnum):
    ASPIRIN = "O=C(C)Oc1ccccc1C(=O)O"
    BENZENE = "c1ccccc1"
    ETHANOL = "OCC"
    MALONALDEHYDE = "O=CCC=O"
    NAPHTHALENE = "c1c2ccccc2ccc1"
    SALICYLIC_ACID = "O=C(O)c1ccccc1O"
    TOLUENE = "Cc1ccccc1"
    URACIL = "O=C1C=CNC(=O)N1"
    ALILINE = "Nc1ccccc1"
    TRIAZOLOPYRAZINE = "C1=NC2=NNN=C2N=C1"
    TROPANE = "N1(C)[C@H]2CC[C@@H]1CCC2"


def get_mol_supplier(filename: str) -> Chem.ForwardSDMolSupplier | None:
    """
    Get a molecule supplier for either gzipped or uncompressed SDF files.

    Args:
        filename: Path to the SDF file (compressed or uncompressed)

    Returns:
        A ForwardSDMolSupplier for the specified file, or None if the file does not exist.
    """
    if not os.path.exists(filename):
        tqdm.write(f"File not found: {filename}")
        return None

    # Check if file is gzipped
    if filename.endswith(".gz"):
        total_size = os.path.getsize(filename)
        non_gz_filename = filename.replace(".gz", "")

        # If uncompressed version already exists, use it directly
        if os.path.exists(non_gz_filename):
            return Chem.ForwardSDMolSupplier(non_gz_filename)

        # Otherwise uncompress the file
        tqdm.write(f"Unzipping {filename}...")
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                with gzip.open(filename, "rb") as f_in:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc="Unzipping",
                        position=1,
                        leave=False,
                    ) as pbar:
                        while True:
                            buf = f_in.read(1024 * 1024)  # 1MB chunks
                            if not buf:
                                break
                            tmp.write(buf)
                            pbar.update(len(buf))
                tmp_filename = tmp.name
            return Chem.ForwardSDMolSupplier(tmp_filename)
        except Exception as e:
            tqdm.write(f"Error unzipping {filename}: {e}")
            # Try to use the file directly without unzipping
            tqdm.write(f"Trying to use {filename} directly...")
            try:
                return Chem.ForwardSDMolSupplier(filename)
            except Exception as e2:
                tqdm.write(f"Error using file directly: {e2}")
                return None

    # For uncompressed files, return supplier directly
    return Chem.ForwardSDMolSupplier(filename)


def analyze_dataset():
    """
    Analyzes the PubChem dataset to find molecules similar to a basis set.
    """
    # --- Configuration ---
    basis_molecule_set = list(MD17Smiles)
    similarity_lower_bound = 0.875
    similarity_upper_bound = 0.925
    allowed_atoms = {"C", "H", "O", "N"}
    max_atoms_of_type = {"O": 5, "N": 3}
    batch_size = 10_000

    # Check if the PubChem directory exists and list some files
    pubchem_dir = "/mnt/d/PubChem"
    if os.path.exists(pubchem_dir):
        tqdm.write(f"PubChem directory exists: {pubchem_dir}")
        try:
            files = os.listdir(pubchem_dir)
            tqdm.write(f"Found {len(files)} files in directory")
            if files:
                tqdm.write(f"First few files: {files[:5]}")
        except Exception as e:
            tqdm.write(f"Error listing directory: {e}")
    else:
        tqdm.write(f"PubChem directory does not exist: {pubchem_dir}")
        return

    # --- Initialization ---
    fp_gen = rdFingerprintGenerator.GetMorganGenerator()
    basis_molecules: list[tuple[MD17Smiles, Chem.Mol, object]] = []
    for basis_enum in basis_molecule_set:
        basis_mol = Chem.MolFromSmiles(basis_enum.value)
        if basis_mol:
            basis_fp = fp_gen.GetFingerprint(basis_mol)
            basis_molecules.append((basis_enum, basis_mol, basis_fp))

    basis_fps = [fp for _, _, fp in basis_molecules]

    total_molecules_processed = 0
    total_similarity = 0.0
    million_satisfying_criteria = 0
    million_counter = 0
    file_counter = 1

    total_compounds_read = 0
    # Track the last million boundary we processed
    last_million_boundary = 0

    # --- Processing Loop ---
    while file_counter <= 100:  # Safety break
        candidate_file = f"/mnt/d/PubChem/Compound_{(file_counter - 1) * 500000 + 1:09d}_{file_counter * 500000:09d}.sdf.gz"

        sup = get_mol_supplier(candidate_file)
        if sup is None:
            tqdm.write(f"Could not open file: {candidate_file}")
            tqdm.write("Trying alternative file naming...")

            # Try alternative naming patterns
            alt_files = [
                f"/mnt/d/PubChem/Compound_{(file_counter - 1) * 500000 + 1:09d}_{file_counter * 500000:09d}.sdf",
                f"/mnt/d/PubChem/compound_{(file_counter - 1) * 500000 + 1:09d}_{file_counter * 500000:09d}.sdf.gz",
            ]

            sup = None
            for alt_file in alt_files:
                sup = get_mol_supplier(alt_file)
                if sup is not None:
                    tqdm.write(f"Found alternative file: {alt_file}")
                    break

            if sup is None:
                tqdm.write("Stopping: Could not open any candidate file.")
                break

        candidate_batch = []
        for mol in tqdm(sup, desc=f"File {file_counter}", unit="mol"):
            total_compounds_read += 1

            if mol is None:
                continue

            # --- Pre-computation criteria ---
            if any(atom.GetSymbol() not in allowed_atoms for atom in mol.GetAtoms()):
                continue
            if any(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element) > count for element, count in max_atoms_of_type.items()):
                continue
            if len(Chem.GetMolFrags(mol)) > 1:
                continue

            candidate_batch.append(mol)

            if len(candidate_batch) >= batch_size:
                # Process the batch
                batch_fps = fp_gen.GetFingerprints(candidate_batch, numThreads=os.cpu_count() or 1)
                for i in range(len(candidate_batch)):
                    candidate_fp = batch_fps[i]

                    similarities = BulkTanimotoSimilarity(candidate_fp, basis_fps)
                    max_similarity = max(similarities) if similarities else 0.0

                    total_similarity += max_similarity
                    total_molecules_processed += 1

                    if similarity_lower_bound <= max_similarity <= similarity_upper_bound:
                        million_satisfying_criteria += 1

                candidate_batch = []  # Reset batch

            # Check if we've crossed a million boundary of TOTAL compounds read
            current_million_boundary = total_compounds_read // 1_000_000
            if current_million_boundary > last_million_boundary:
                million_counter += 1
                mean_similarity = (total_similarity / total_molecules_processed) if total_molecules_processed > 0 else 0.0
                print(f"\n--- Stats for million #{million_counter} (total compounds read) ---")
                print(f"Mean similarity to seeds (for {total_molecules_processed} filtered compounds): {mean_similarity:.4f}")
                print(f"Molecules satisfying criteria in this million-chunk: {million_satisfying_criteria}")
                print("---------------------------\n")
                million_satisfying_criteria = 0
                last_million_boundary = current_million_boundary

        # Process any remaining molecules in the batch at the end of the file
        if candidate_batch:
            batch_fps = fp_gen.GetFingerprints(candidate_batch, numThreads=os.cpu_count() or 1)
            for i in range(len(candidate_batch)):
                candidate_fp = batch_fps[i]

                similarities = BulkTanimotoSimilarity(candidate_fp, basis_fps)
                max_similarity = max(similarities) if similarities else 0.0

                total_similarity += max_similarity
                total_molecules_processed += 1

                if similarity_lower_bound <= max_similarity <= similarity_upper_bound:
                    million_satisfying_criteria += 1

        file_counter += 1

    print("Analysis complete.")


if __name__ == "__main__":
    analyze_dataset()
