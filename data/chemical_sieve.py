from rdkit import Chem, RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from collections import defaultdict
import gzip
import tempfile
import os
from enum import StrEnum
import numpy as np

RDLogger.DisableLog("rdApp.*")  # type: ignore


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


rmd17_molecules = [
    "Benzene",
    "Ethanol",
    "Malonaldehyde",
    "Naphthalene",
    "Paracetamol",
    "Salicylic acid",
    "Toluene",
    "Uracil",
]

MD61_molecules = [
    # MD17 molecules
    "Aspirin",
    "Benzene",
    "Ethanol",
    "Malonaldehyde",
    "Naphthalene",
    "Salicylic acid",
    "Toluene",
    "Uracil",
    # Our choices below
    "Paracetamol",
    "Caffeine",
]

candidate_results = {
    "mol_generation_failed": 0,
    "too_large": 0,
    "no_smiles": 0,
    "invalid_atoms": 0,
    "multiple_fragments": 0,
    "too_planar": 0,
    "valid": 0,
    "too_similar_to_existing": 0,
    "too_similar_to_other_sets": 0,
}

max_atoms_of_type = {
    "O": 5,
    "N": 3,
}


def get_mol_supplier(filename: str) -> Chem.ForwardSDMolSupplier:
    """
    Get a molecule supplier for either gzipped or uncompressed SDF files.

    Args:
        filename: Path to the SDF file (compressed or uncompressed)

    Returns:
        A ForwardSDMolSupplier for the specified file
    """
    # Check if file is gzipped
    if filename.endswith(".gz"):
        total_size = os.path.getsize(filename)
        non_gz_filename = filename.replace(".gz", "")

        # If uncompressed version already exists, use it directly
        if os.path.exists(non_gz_filename):
            return Chem.ForwardSDMolSupplier(non_gz_filename)

        # Otherwise uncompress the file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with gzip.open(filename, "rb") as f_in:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Unzipping",
                    position=1,
                ) as pbar:
                    while True:
                        buf = f_in.read(1024 * 1024)  # 1MB chunks
                        if not buf:
                            break
                        tmp.write(buf)
                        pbar.update(len(buf))
            tmp_filename = tmp.name
        return Chem.ForwardSDMolSupplier(tmp_filename)

    # For uncompressed files, return supplier directly
    return Chem.ForwardSDMolSupplier(filename)


def process_candidate_file(
    candidate_file: str,
    basis_molecules: list[tuple[MD17Smiles, Chem.Mol, ExplicitBitVect]],
    fp_gen: rdFingerprintGenerator.FingerprintGenerator64,
    similarity_lower_bound: float,
    similarity_upper_bound: float,
    allowed_atoms: set[str],
    seen_smiles_dict: dict[MD17Smiles, set[str]],
    similar_molecules_dict: dict[MD17Smiles, list[Chem.Mol]],
    candidates_per_molecule: int,
    batch_size: int = 10_000,
    max_internal_similarity: float = 0.95,
    max_inter_set_similarity: float = 0.8,
    stats: defaultdict[str, float] | None = None,
) -> tuple[dict[MD17Smiles, list[Chem.Mol]], defaultdict[str, int]]:
    """
    Process a single candidate file for multiple basis molecules.

    Args:
        candidate_file: Path to the candidate file
        basis_molecules: List of tuples with (enum_member, molecule, fingerprint)
        fp_gen: Fingerprint generator
        similarity_lower_bound: Minimum similarity threshold to basis molecule
        similarity_upper_bound: Maximum similarity threshold to basis molecule
        allowed_atoms: Set of allowed atom types
        seen_smiles_dict: Dictionary mapping basis molecule enum members to sets of seen SMILES
        similar_molecules_dict: Dictionary mapping basis molecule enum members to lists of similar molecules
        candidates_per_molecule: Number of similar molecules to find per basis molecule
        batch_size: Size of batches for processing molecules
        max_internal_similarity: Maximum allowed similarity between any two molecules in the result set
        max_inter_set_similarity: Maximum allowed similarity between any molecule in the result set and any molecule in all other sets
        stats: A dictionary to track processing statistics.

    Returns:
        Updated similar_molecules_dict and candidate_results
    """
    candidate_results: defaultdict[str, int] = defaultdict(int)
    candidate_batch: list[Chem.Mol] = []

    if stats is None:
        stats = defaultdict(float)

    # Store fingerprints of ALL molecules across ALL sets
    all_existing_fps: list[ExplicitBitVect] = []
    all_existing_mols_info: list[
        tuple[MD17Smiles, int]
    ] = []  # Store (enum, index) for each fingerprint

    # Store fingerprints of molecules we've already added to each set
    existing_fps_dict: dict[MD17Smiles, list[ExplicitBitVect]] = {}
    for basis_enum, molecules in similar_molecules_dict.items():
        if molecules:
            fps = fp_gen.GetFingerprints(molecules)
            all_existing_fps.extend(fps)
            all_existing_mols_info.extend([(basis_enum, i) for i in range(len(fps))])
            existing_fps_dict[basis_enum] = fps
        else:
            existing_fps_dict[basis_enum] = []

    try:
        sup = get_mol_supplier(candidate_file)
    except Exception as e:
        tqdm.write(f"Error unzipping {candidate_file}: {e}")
        return similar_molecules_dict, candidate_results

    # Keep track of which basis molecules still need more candidates
    active_basis_molecules = [
        bm
        for bm in basis_molecules
        if len(similar_molecules_dict[bm[0]]) < candidates_per_molecule
    ]

    if not active_basis_molecules:
        return similar_molecules_dict, candidate_results

    all_basis_fps = [bm[2] for bm in basis_molecules]

    for mol in tqdm(sup, total=500000):
        stats["total_compounds_read"] += 1

        mol: Chem.Mol | None
        if mol is None:
            candidate_results["mol_generation_failed"] += 1
            continue

        basis_num_atoms = active_basis_molecules[0][
            1
        ].GetNumHeavyAtoms()  # Use first molecule as reference

        if (
            mol.GetNumHeavyAtoms() > basis_num_atoms * 1
            or mol.GetNumHeavyAtoms() < basis_num_atoms * 0.2
        ):
            candidate_results["too_large"] += 1
        elif mol.GetNumAtoms() == 0:
            candidate_results["no_atoms"] += 1
        elif any(atom.GetSymbol() not in allowed_atoms for atom in mol.GetAtoms()):
            candidate_results["invalid_atoms"] += 1

        # Check if any element exceeds its maximum allowed count
        elif any(
            sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == element) > count
            for element, count in max_atoms_of_type.items()
            if element in allowed_atoms
        ):
            candidate_results["too_many_atoms_of_type"] += 1
        elif len(Chem.GetMolFrags(mol)) > 1:
            candidate_results["multiple_fragments"] += 1
        else:
            candidate_results["valid"] += 1
            candidate_batch.append(mol)

        if len(candidate_batch) >= batch_size:
            # This logic is intentionally reverted to a previous, flawed version
            # to accurately simulate and analyze an existing, already-generated dataset.
            batch_fps = fp_gen.GetFingerprints(candidate_batch, numThreads=5)
            new_fps_in_this_batch: list[ExplicitBitVect] = []

            # --- General stats calculation (for mean similarity) ---
            for i in range(len(candidate_batch)):
                candidate_fp = batch_fps[i]
                similarities_to_all_basis = BulkTanimotoSimilarity(
                    candidate_fp, all_basis_fps
                )
                in_range_similarities = [
                    s
                    for s in similarities_to_all_basis
                    if similarity_lower_bound <= s <= similarity_upper_bound
                ]
                if in_range_similarities:
                    relevant_similarity = max(in_range_similarities)
                elif similarities_to_all_basis:
                    relevant_similarity = max(similarities_to_all_basis)
                else:
                    relevant_similarity = 0.0
                stats["total_similarity"] += relevant_similarity
                stats["total_molecules_processed_for_similarity"] += 1

            # --- Flawed selection logic simulation ---
            for basis_enum, basis_mol, basis_fp in active_basis_molecules:
                remaining = candidates_per_molecule - len(
                    similar_molecules_dict[basis_enum]
                )
                if remaining <= 0:
                    continue

                similarities = BulkTanimotoSimilarity(
                    basis_fp, batch_fps, returnDistance=True
                )
                sim_with_idx = [(sim, idx) for idx, sim in enumerate(similarities)]
                sim_with_idx.sort(reverse=False)

                for sim, idx in sim_with_idx:
                    if remaining <= 0:
                        break

                    if similarity_lower_bound <= sim <= similarity_upper_bound:
                        mol_smiles = Chem.MolToSmiles(candidate_batch[idx])
                        if mol_smiles not in seen_smiles_dict[basis_enum]:
                            candidate_fp = batch_fps[idx]

                            if existing_fps_dict[basis_enum]:
                                internal_sims = BulkTanimotoSimilarity(
                                    candidate_fp, existing_fps_dict[basis_enum]
                                )
                                if max(internal_sims) > max_internal_similarity:
                                    continue

                            # The OLD, FLAWED check is performed here for selection
                            old_check_passed = True
                            if all_existing_fps:
                                external_sims = BulkTanimotoSimilarity(
                                    candidate_fp, all_existing_fps
                                )
                                if max(external_sims) > max_inter_set_similarity:
                                    old_check_passed = False

                            if old_check_passed:
                                # The NEW, CORRECT check is performed here just to count errors
                                correct_check_fps = (
                                    all_existing_fps + new_fps_in_this_batch
                                )
                                if correct_check_fps:
                                    correct_external_sims = BulkTanimotoSimilarity(
                                        candidate_fp, correct_check_fps
                                    )
                                    if (
                                        max(correct_external_sims)
                                        > max_inter_set_similarity
                                    ):
                                        stats["incorrectly_added_molecules"] += 1

                                # Add molecule based on old logic to simulate dataset generation
                                stats["total_satisfying_criteria"] += 1
                                similar_molecules_dict[basis_enum].append(
                                    candidate_batch[idx]
                                )
                                seen_smiles_dict[basis_enum].add(mol_smiles)
                                existing_fps_dict[basis_enum].append(candidate_fp)
                                new_fps_in_this_batch.append(candidate_fp)
                                remaining -= 1

            all_existing_fps.extend(new_fps_in_this_batch)
            candidate_batch = []
            active_basis_molecules = [
                bm
                for bm in basis_molecules
                if len(similar_molecules_dict[bm[0]]) < candidates_per_molecule
            ]

        # Check if we've crossed a 100k boundary of TOTAL compounds read
        current_boundary = stats["total_compounds_read"] // 100_000
        if int(current_boundary) > int(stats["last_boundary"]):
            mean_similarity = (
                (
                    stats["total_similarity"]
                    / stats["total_molecules_processed_for_similarity"]
                )
                if stats["total_molecules_processed_for_similarity"] > 0
                else 0.0
            )
            tqdm.write("\n---")
            tqdm.write(
                f"Total satisfying criteria (old logic): {int(stats['total_satisfying_criteria'])}"
            )
            tqdm.write(
                f"Incorrectly added molecules (due to flaw): {int(stats['incorrectly_added_molecules'])}"
            )
            tqdm.write(f"Mean similarity of all compounds seen: {mean_similarity:.4f}")
            tqdm.write(f"Total compounds seen: {int(stats['total_compounds_read'])}")
            tqdm.write("---\n")
            stats["last_boundary"] = current_boundary

    return similar_molecules_dict, candidate_results


def planarity_score(mol: Chem.Mol) -> float:
    """
    Higher score means more planar.
    """
    conf = mol.GetConformer()
    # Extract 3D coordinates
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    # Center coordinates
    coords_centered = coords - coords.mean(axis=0)
    # Perform SVD; the smallest singular value corresponds to deviation from planarity
    _, s, vh = np.linalg.svd(coords_centered)
    normal = vh[-1]  # best-fit plane normal
    # Compute distances from the plane
    distances = np.abs(coords_centered.dot(normal))
    rmsd = np.sqrt(np.mean(distances**2))
    return rmsd


def generate_similar_dataset(
    basis_molecule_set: list[MD17Smiles],
    candidates_per_molecule: int,
    similarity_lower_bound: float,
    similarity_upper_bound: float,
    allowed_atoms: set[str],
    max_internal_similarity: float,
    max_inter_set_similarity: float,
    batch_size: int = 10_000,
) -> dict[MD17Smiles, list[Chem.Mol]]:
    """
    Generate a dataset of similar molecules to the basis molecules.

    Args:
        basis_molecule_set: Set of basis molecules to find similar molecules for
        candidates_per_molecule: Number of similar molecules to find per basis molecule
        similarity_lower_bound: Minimum similarity threshold to basis molecule
        similarity_upper_bound: Maximum similarity threshold to basis molecule
        allowed_atoms: Set of allowed atom types
        batch_size: Size of batches for processing molecules
        max_internal_similarity: Maximum allowed similarity between any two molecules in the result set
        max_inter_set_similarity: Maximum allowed similarity between any molecule in the result set and any molecule in all other sets

    Returns:
        Dictionary mapping basis molecule enum members to lists of similar molecules
    """
    # Initialize data structures
    similar_molecules_dict: dict[MD17Smiles, list[Chem.Mol]] = {}
    seen_smiles_dict: dict[MD17Smiles, set[str]] = {}
    stats = defaultdict(float)

    # Prepare basis molecules and fingerprints
    fp_gen = rdFingerprintGenerator.GetMorganGenerator()
    basis_molecules: list[tuple[MD17Smiles, Chem.Mol, ExplicitBitVect]] = []

    for basis_enum in basis_molecule_set:
        basis_mol = Chem.MolFromSmiles(basis_enum)
        if basis_mol is None:
            raise ValueError(
                f"Failed to generate basis molecule from SMILES: {basis_enum}"
            )

        basis_fp = fp_gen.GetFingerprint(basis_mol)
        basis_molecules.append((basis_enum, basis_mol, basis_fp))

        # Initialize result containers
        similar_molecules_dict[basis_enum] = []
        seen_smiles_dict[basis_enum] = set()

    # Process files until we have enough molecules or run out of files
    file_counter = 1

    while True:
        # Check if we're done for all molecules
        all_complete = all(
            len(mols) >= candidates_per_molecule
            for mols in similar_molecules_dict.values()
        )
        if all_complete:
            tqdm.write("Found all required similar molecules for all basis molecules")
            break

        candidate_file = f"/mnt/d/PubChem/Compound_{(file_counter - 1) * 500000 + 1:09d}_{file_counter * 500000:09d}.sdf.gz"
        file_counter += 1

        similar_molecules_dict, candidate_results = process_candidate_file(
            candidate_file,
            basis_molecules,
            fp_gen,
            similarity_lower_bound,
            similarity_upper_bound,
            allowed_atoms,
            seen_smiles_dict,
            similar_molecules_dict,
            candidates_per_molecule,
            batch_size,
            max_internal_similarity,
            max_inter_set_similarity,
            stats=stats,
        )

        # Use enum.name to display molecule names
        molecules_found = [(k.name, len(v)) for k, v in similar_molecules_dict.items()]
        tqdm.write(f"Processed file {file_counter - 1}, {candidate_file}")
        tqdm.write(f"Molecules found so far: {molecules_found}")

        if file_counter > 100:  # Safety limit
            tqdm.write(
                "Reached file limit. Some molecules may not have enough similar candidates."
            )
            break

    # Ensure we have exactly candidates_per_molecule results
    for basis_enum in similar_molecules_dict:
        if len(similar_molecules_dict[basis_enum]) > candidates_per_molecule:
            tqdm.write(f"Trimming excess molecules for {basis_enum.name}")
            similar_molecules_dict[basis_enum] = similar_molecules_dict[basis_enum][
                :candidates_per_molecule
            ]

    return similar_molecules_dict


def pretty_print_similar_molecules_dict(
    similar_molecules_dict: dict[MD17Smiles, list[Chem.Mol]],
) -> None:
    for basis_enum, similar_molecules in similar_molecules_dict.items():
        tqdm.write(f"{basis_enum.name}:")
        for mol in similar_molecules:
            tqdm.write(f"  {Chem.MolToSmiles(mol)}")


if __name__ == "__main__":
    from rdkit.Chem import Draw
    import os

    similar_molecules_dict = generate_similar_dataset(
        basis_molecule_set=list(MD17Smiles),
        candidates_per_molecule=99999,
        similarity_lower_bound=0.875,
        similarity_upper_bound=0.925,
        allowed_atoms=set(["C", "H", "O", "N"]),
        max_internal_similarity=0.2,
        max_inter_set_similarity=0.2,
    )

    # Write results to a formatted text file
    with open("data/similar_molecules_results.txt", "w") as f:
        f.write("Similar Molecules Dataset Results\n")
        f.write("===============================\n\n")

        for basis_enum, similar_molecules in similar_molecules_dict.items():
            f.write(f"{basis_enum.name}:\n")
            f.write("-" * (len(basis_enum.name) + 1) + "\n")
            for i, mol in enumerate(similar_molecules, 1):
                f.write(f"  {i}. {Chem.MolToSmiles(mol)}\n")
            f.write("\n")

        f.write("\nSummary:\n")
        f.write("--------\n")
        for basis_enum, similar_molecules in similar_molecules_dict.items():
            f.write(f"{basis_enum.name}: {len(similar_molecules)} molecules\n")

    print("Results written to similar_molecules_results.txt")

    # Create directory for images if it doesn't exist
    os.makedirs("data/molecule_images", exist_ok=True)

    # Create a big tiled image with all molecules
    all_mols = []
    all_legends = []

    for basis_enum, similar_molecules in similar_molecules_dict.items():
        basis_mol = Chem.MolFromSmiles(basis_enum.value)
        all_mols.append(basis_mol)
        all_legends.append(f"{basis_enum.name} (Original)")

        for i, mol in enumerate(similar_molecules):
            all_mols.append(mol)
            all_legends.append(f"{basis_enum.name} Similar {i + 1}")

    # Draw all molecules in one big grid
    big_img = Draw.MolsToGridImage(
        all_mols, molsPerRow=8, subImgSize=(200, 200), legends=all_legends, useSVG=False
    )

    # Save the big image
    big_img_path = "data/molecule_images/all_molecules.png"
    big_img.save(big_img_path)
    print(f"Big tiled image with all molecules saved to {big_img_path}")
