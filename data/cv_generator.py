import random
from atom.training.config_options import TG80MoleculeType
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import numpy.typing as npt
import os
import umap
from sklearn.cluster import AgglomerativeClustering


def _safe_disable_rdlogger() -> None:
    func = getattr(RDLogger, "DisableLog", None)
    if callable(func):
        try:
            func("rdApp.*")  # type: ignore[arg-type]
        except Exception:
            pass


_ = _safe_disable_rdlogger()

BASE_SEED = 42
molecules = [m.value for m in TG80MoleculeType]

tg_80_smiles = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Toluene": "Cc1ccccc1",
    "Uracil": "O=C1C=CNC(=O)N1",
    "Salicylic acid": "OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ethanol": "CCO",
    "Malonaldehyde": "O=CC=C=O",
    "Benzene": "c1ccccc1",
    "Azobenzene": "c1ccc(N=Nc2ccccc2)cc1",
    "Thymine": "CC1=CN(C(=O)NC1=O)",
    "Octanol": "CCCCCCCCO",
    "Naphthalene": "c1ccc2ccccc2c1",
    "Anthracene": "c1ccc2cc3ccccc3cc2c1",
    "Phenanthrene": "c1ccc2c(c1)C=CC3=CC=CC=C23",
    "Biphenyl": "c1ccc(cc1)c2ccccc2",
    "Styrene": "C=Cc1ccccc1",
    "Anisole": "COc1ccccc1",
    "Chlorobenzene": "Clc1ccccc1",
    "Nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "p-Cresol": "Cc1ccc(O)cc1",
    "p-Xylene": "Cc1ccc(C)cc1",
    "Pyridine": "c1ccncc1",
    "Pyrimidine": "c1cncnc1",
    "Imidazole": "c1cnc[nH]1",
    "Furan": "c1ccoc1",
    "Thiophene": "c1ccsc1",
    "Indole": "c1ccc2c(c1)[nH]c(c2)",
    "Quinoline": "c1ccc2ncccc2c1",
    "Isoquinoline": "C1(C=NC=C2)=C2C=CC=C1",
    "Purine": "c1c2c(nc[nH]2)ncn1",
    "Coumarin": "O=C1Oc2ccccc2C=C1",
    "Benzoicacid": "c1ccc(cc1)C(=O)O",
    "Aceticacid": "CC(=O)O",
    "Formicacid": "C(=O)O",
    "Propionicacid": "CCC(=O)O",
    "Butyricacid": "CCCC(=O)O",
    "Oxalicacid": "OC(=O)C(=O)O",
    "Malonicacid": "OC(=O)C(C(=O)O)",
    "Succinicacid": "O=C(O)CC(=O)O",
    "Tartaricacid": "OC(=O)[C@H](O)[C@H](O)C(=O)O",
    "Citricacid": "C(C(=O)O)C(O)(C(=O)O)CC(=O)O",
    "Lacticacid": "CC(O)C(=O)O",
    "Methanol": "CO",
    "Isopropanol": "CC(C)O",
    "Butanol": "CCCCO",
    "Pentanol": "CCCCCO",
    "Hexanol": "CCCCCCO",
    "Heptanol": "CCCCCCCO",
    "Decanol": "CCCCCCCCCO",
    "Cyclohexanol": "C1CCCCC1O",
    "Aniline": "c1ccccc1N",
    "Dimethylaniline": "CN(C)c1ccccc1",
    "Trimethylamine": "N(C)(C)C",
    "Ethylamine": "CCN",
    "Propylamine": "CCCN",
    "Butylamine": "CCCCN",
    "Benzylamine": "c1ccccc1CN",
    "Formamide": "C(=O)N",
    "Acetamide": "CC(=O)N",
    "Acetone": "CC(=O)C",
    "Acetaldehyde": "CC=O",
    "Benzaldehyde": "c1ccccc1C=O",
    "Cyclohexanone": "O=C1CCCCC1",
    "Cyclopentanone": "O=C1CCCC1",
    "Furfural": "O=Cc1ccco1",
    "Formaldehyde": "C=O",
    "2-Butanone": "CC(=O)CC",
    "Ethylene": "C=C",
    "Propylene": "C=CC",
    "1.3-Butadiene": "C=CC=C",
    "1.3.5-Hexatriene": "C=CC=CC=C",
    "Ethylene oxide": "C1CO1",
    "Propylene oxide": "CC1CO1",
    "Cyclopropane": "C1CC1",
    "Cyclobutane": "C1CCC1",
    "Cyclopentadiene": "C1=CC=CC1",
    "1.3-Cyclohexadiene": "C1=CC=CCC1",
    "Dimethylsulfide": "CSC",
    "Ethanethiol": "CCS",
    "Benzothiophene": "s2c1ccccc1cc2",
    "Dichloromethane": "ClCCl",
    "Chloroform": "ClC(Cl)Cl",
    "Carbon tetrachloride": "C(Cl)(Cl)(Cl)Cl",
    "Tetrachloroethene": "ClC(Cl)=C(Cl)Cl",
    "1.2-Dichloroethane": "ClCCCl",
    "Cyclohexane": "C1CCCCC1",
    "Decane": "CCCCCCCCCC",
    "Propane": "CCC",
    "Butane": "CCCC",
    "Isobutane": "CC(C)C",
    "Tetrahydrofuran": "C1CCOC1",
    "1.4-Dioxane": "C1COCCO1",
    "Acetonitrile": "CC#N",
    "N.N-Dimethylformamide": "CN(C)C=O",
    "malondialdehyde1": "O=CNc1ccccc1C=O",
    "toluene1": "OCc1cccc(O)c1",
    "TRIAZOLOPYRAZINE": "c1ccc(-c2ccn[nH]2)cc1",
    "benzene1": "Nc1ccccc1C(=O)O",
    "benzene2": "O=C(O)c1ccccn1",
    "ethanol1": "Cc1ncc(CO)c(CO)c1O",
    "ethanol2": "OCc1cnc[nH]1",
    "ethanol3": "OCCc1ccccc1",
    "malondialdehyde2": "O=C1C=CC(=O)C=C1",
    "Salicylicacid1": "NOCC(=O)O",
    "Salicylicacid2": "O=C(O)C(O)Cc1cnc[nH]1",
    "Salicylicacid3": "COc1cc(CCN)ccc1O",
    "toluene2": "O=c1[nH]c2ccccc2c2ccccc12",
    "uracil1": "O=C1CC(=O)NC(=O)N1",
    "uracil2": "c1cc2ccc3cccc4[nH]c(c1)c2c34",
    "tropane1": "O=C(O)C1CCCCC1",
    "tropane2": "CCCCN1CC1",
    "tropane3": "CC1CO1",
}
print(f"Length of tg_80_smiles: {len(tg_80_smiles)}")


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


def _normalize_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _build_normalized_smiles_map(raw_map: dict[str, str]) -> dict[str, str]:
    norm_map: dict[str, str] = {}
    for k, v in raw_map.items():
        nk = _normalize_key(k)
        if nk in norm_map and norm_map[nk] != v:
            raise ValueError(f"Duplicate normalized key with different SMILES: {k}")
        norm_map[nk] = v
    return norm_map


def _names_to_smiles(names: list[str], raw_map: dict[str, str]) -> list[str]:
    norm_map = _build_normalized_smiles_map(raw_map)
    smiles_list: list[str] = []
    missing: list[str] = []
    for n in names:
        key = _normalize_key(n)
        smi = norm_map.get(key)
        if smi is None:
            missing.append(n)
        else:
            smiles_list.append(smi)
    if missing:
        raise ValueError(f"Missing SMILES for: {', '.join(missing)}")
    return smiles_list


def _compute_ecfp4_dense(smiles: list[str], fp_size: int = 512) -> npt.NDArray[np.float32]:
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    X = np.zeros((len(smiles), fp_size), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {s}")
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr
    return X


def _assign_clusters_to_folds(cluster_labels: list[int], names: list[str], n_folds: int) -> dict[int, list[str]]:
    clusters: dict[int, list[str]] = {}
    for name, lab in zip(names, cluster_labels):
        clusters.setdefault(int(lab), []).append(name)

    fold_to_names: dict[int, list[str]] = {i: [] for i in range(n_folds)}
    fold_sizes: list[int] = [0] * n_folds

    for _, members in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
        target_fold = int(min(range(n_folds), key=lambda i: fold_sizes[i]))
        fold_to_names[target_fold].extend(members)
        fold_sizes[target_fold] += len(members)

    return fold_to_names


def write_umap_cluster_folds(n_folds: int = 5) -> None:

    os.makedirs("data/crossval_folds_umap", exist_ok=True)

    names = molecules.copy()
    # Map names (enum values) to SMILES via normalization across the provided dictionary
    smiles = _names_to_smiles(names, tg_80_smiles)
    X = _compute_ecfp4_dense(smiles, fp_size=512)

    reducer = umap.UMAP(n_components=32, random_state=BASE_SEED, n_neighbors=15, min_dist=0.1, metric="jaccard")
    emb = reducer.fit_transform(X)

    ward = AgglomerativeClustering(n_clusters=n_folds * 2, linkage="ward")
    cluster_labels = ward.fit_predict(emb).tolist()

    fold_to_names = _assign_clusters_to_folds(cluster_labels, names, n_folds)

    for test_fold in range(n_folds):
        val_fold = (test_fold + 1) % n_folds
        test = sorted(fold_to_names[test_fold])
        val = sorted(fold_to_names[val_fold])
        train = sorted([n for f, ns in fold_to_names.items() if f not in [test_fold, val_fold] for n in ns])
        filename = f"data/crossval_folds_umap/fold{test_fold + 1}.toml"
        write_toml_split(filename, train, val, test)
        print(f"Wrote {filename}: train={len(train)}, val={len(val)}, test={len(test)}")


# Shuffle molecules once with a fixed seed (random baseline)
random.seed(BASE_SEED)
shuffled = molecules.copy()
random.shuffle(shuffled)

# Calculate fold size and number of folds using array_split to handle remainders
n_molecules = len(shuffled)
n_folds = 5
idxs = np.arange(n_molecules)
fold_idxs = np.array_split(idxs, n_folds)

for fold in range(n_folds):
    test_idx = fold_idxs[fold]
    val_idx = fold_idxs[(fold + 1) % n_folds]
    train_idx = np.concatenate([fold_idxs[f] for f in range(n_folds) if f not in [fold, (fold + 1) % n_folds]])

    test = [shuffled[i] for i in test_idx]
    val = [shuffled[i] for i in val_idx]
    train = [shuffled[i] for i in train_idx]

    filename = f"data/crossval_folds/fold{fold + 1}.toml"
    write_toml_split(filename, train, val, test)
    print(f"Wrote {filename}: train={len(train)}, val={len(val)}, test={len(test)}")

# Also write UMAP-based folds
write_umap_cluster_folds(n_folds=5)
