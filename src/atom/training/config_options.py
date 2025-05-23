from enum import StrEnum
from typing import final


@final
class DataPartition(StrEnum):
    train = "train"
    val = "val"
    test = "test"


@final
class Datasets(StrEnum):
    md17 = "md17"
    rmd17 = "rmd17"
    tg80 = "tg80"
    nbody_simple = "nbody_simple"


@final
class MD17MoleculeType(StrEnum):
    aspirin = "aspirin"
    benzene = "benzene"
    ethanol = "ethanol"
    malonaldehyde = "malonaldehyde"
    naphthalene = "naphthalene"
    salicylic = "salicylic"
    toluene = "toluene"
    uracil = "uracil"


@final
class RMD17MoleculeType(StrEnum):
    azobenzene = "azobenzene"
    benzene = "benzene"
    ethanol = "ethanol"
    malonaldehyde = "malonaldehyde"
    naphthalene = "naphthalene"
    paracetamol = "paracetamol"
    salicylic = "salicylic"
    toluene = "toluene"
    uracil = "uracil"


@final
class TG80MoleculeType(StrEnum):
    acetaldehyde = "acetaldehyde"
    acetamide = "acetamide"
    aceticacid = "aceticacid"  # NEW
    acetonitrile = "acetonitrile"
    aniline = "aniline"
    anthracene = "anthracene"
    aspirin = "aspirin"
    benzaldehyde = "benzaldehyde"
    benzene = "benzene"
    benzene1 = "benzene1"
    benzene2 = "benzene2"
    benzoicacid = "benzoicacid"  # Test
    benzothiophene = "benzothiophene"
    benzylamine = "benzylamine"
    biphenyl = "biphenyl"
    butane = "butane"
    butanol = "butanol"
    butylamine = "butylamine"
    butadiene = "1.3-butadiene"
    butanone = "2-butanone"
    chlorobenzene = "chlorobenzene"
    chloroform = "chloroform"
    citricacid = "citricacid"
    coumarin = "coumarin"
    cyclobutane = "cyclobutane"
    cyclohexadiene = "1.3-cyclohexadiene"
    cyclohexane = "cyclohexane"
    cyclohexanol = "cyclohexanol"
    cyclohexanone = "cyclohexanone"
    cyclopentadiene = "cyclopentadiene"
    cyclopentanone = "cyclopentanone"
    cyclopropane = "cyclopropane"
    dichloroethane = "1.2-dichloroethane"
    dioxane = "1.4-dioxane"
    ethanethiol = "ethanethiol"
    ethanol = "ethanol"
    ethanol1 = "ethanol1"
    ethylamine = "ethylamine"
    ethylene = "ethylene"
    formaldehyde = "formaldehyde"
    formamide = "formamide"
    formicacid = "formicacid"
    furan = "furan"
    furfural = "furfural"  # Test
    heptanol = "heptanol"
    hexanol = "hexanol"
    imidazole = "imidazole"
    indole = "indole"
    isobutane = "isobutane"
    isopropanol = "isopropanol"
    isoquinoline = "isoquinoline"
    malondialdehyde1 = "malondialdehyde1"
    malondialdehyde2 = "malondialdehyde2"
    malonicacid = "malonicacid"
    methanol = "methanol"
    naphthalene = "naphthalene"
    nitrobenzene = "nitrobenzene"
    oxalicacid = "oxalicacid"
    paracetamol = "paracetamol"
    pentanol = "pentanol"
    p_cresol = "p-cresol"
    propane = "propane"
    propylene = "propylene"
    pxylene = "p-xylene"
    pyrimidine = "pyrimidine"
    quinoline = "quinoline"
    salicylicacid1 = "salicylicacid1"
    salicylicacid2 = "salicylicacid2"
    salicylicacid3 = "salicylicacid3"
    styrene = "styrene"
    succinicacid = "succinicacid"
    tetrahydrofuran = "tetrahydrofuran"
    thymine = "thymine"
    toluene = "toluene"
    trimethylamine = "trimethylamine"
    tropane1 = "tropane1"
    tropane2 = "tropane2"
    tropane3 = "tropane3"
    uracil = "uracil"  # MD17, test
    uracil1 = "uracil1"


@final
class OptimizerType(StrEnum):
    SGD = "sgd"
    ADAMW = "adamw"
    ADAM = "adam"
    MUON = "muon"
    ADAM_MINI = "adam-mini"


@final
class SchedulerType(StrEnum):
    NONE = "none"
    STEP = "step"
    COS_ANNEALING = "cosine_annealing"


@final
class ModelType(StrEnum):
    ATOM = "atom"
    EGNO = "EGNO"


@final
class NormType(StrEnum):
    LAYER = "layer"
    RMS = "rms"


@final
class ValueResidualType(StrEnum):
    NONE = "none"
    FIXED = "fixed"
    LEARNABLE = "learnable"


@final
class AttentionType(StrEnum):
    SELF = "self"
    GHCA = "ghca"


@final
class EquivariantLiftingType(StrEnum):
    NONE = "none"
    EQUIVARIANT = "equivariant"
    NO_TP = "no_tensor_product"


@final
class FFNActivation(StrEnum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    RELU2 = "relu2"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
