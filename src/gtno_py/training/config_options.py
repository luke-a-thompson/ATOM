from enum import StrEnum
from typing import final


@final
class DataPartition(StrEnum):
    train = "train"
    val = "val"
    test = "test"


@final
class MD17Version(StrEnum):
    md17 = "md17"
    rmd17 = "rmd17"
    tg80 = "tg80"


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
    acetonitrile = "acetonitrile"
    acetamide = "acetamide"
    acetaldehyde = "acetaldehyde"
    aniline = "aniline"
    anthracene = "anthracene"
    benzaldehyde = "benzaldehyde"
    benzene = "benzene"
    benzene2 = "benzene2"
    benzoicacid = "benzoicacid"  # Test
    benzothiophene = "benzothiophene"
    benzylamine = "benzylamine"
    biphenyl = "biphenyl"
    butane = "butane"
    butadiene = "1.3-butadiene"
    butanone = "2-butanone"
    butanol = "butanol"
    butylamine = "butylamine"
    chloroform = "chloroform"
    chlorobenzene = "chlorobenzene"
    coumarin = "coumarin"
    cyclohexadiene = "1.3-cyclohexadiene"
    cyclohexane = "cyclohexane"
    cyclohexanol = "cyclohexanol"
    cyclohexanone = "cyclohexanone"
    cyclopropane = "cyclopropane"
    dichloroethane = "1.2-dichloroethane"
    ethanethiol = "ethanethiol"
    ethylene = "ethylene"
    ethylamine = "ethylamine"
    formaldehyde = "formaldehyde"
    formamide = "formamide"
    furan = "furan"
    furfural = "furfural"  # Test
    heptanol = "heptanol"
    indole = "indole"
    isopropanol = "isopropanol"
    malondialdehyde1 = "malondialdehyde1"
    malonicacid = "malonicacid"
    methanol = "methanol"
    naphthalene = "naphthalene"
    nitrobenzene = "nitrobenzene"
    oxalicacid = "oxalicacid"
    pentanol = "pentanol"
    p_cresol = "p-cresol"
    propane = "propane"
    propylene = "propylene"
    quinoline = "quinoline"
    salicylicacid2 = "salicylicacid2"
    succinicacid = "succinicacid"
    trimethylamine = "trimethylamine"
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
    GTNO = "GTNO"
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
