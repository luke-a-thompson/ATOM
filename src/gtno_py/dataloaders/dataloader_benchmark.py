from torch.utils.data import DataLoader
import numpy as np
from gtno_py.dataloaders.egno_dataloder import MD17Dataset, MD17DynamicsDataset, MD17MoleculeType, MD17Version, DataPartition

if __name__ == "__main__":
    import time
    from tqdm import tqdm

    # Test MD17Dataset
    dataset_static = MD17Dataset(
        partition=DataPartition.train,
        max_samples=5000,
        delta_frame=30000,
        data_dir="data/",
        split_dir="data/",
        md17_version=MD17Version.rmd17,
        molecule_type=MD17MoleculeType.benzene,
        force_regenerate=False,
        num_timesteps=1,  # Set num_timesteps for replication
    )
    dataloader_static = DataLoader(dataset_static, batch_size=100, shuffle=True)
    # print("MD17Dataset Output Shapes:")
    # for data in dataloader_static:
    #     for key in data:
    #         if key not in ["cfg", "edge_attr"]:
    #             print(f"  {key}:", data[key].shape)
    #     if "cfg" in data:
    #         print("  cfg shapes:")
    #         for key in data["cfg"]:
    #             print(f"    {key}:", data["cfg"][key].shape)
    #     break

    # Test MD17DynamicsDataset
    dataset_dynamic = MD17DynamicsDataset(
        partition=DataPartition.train,
        max_samples=500,
        delta_frame=3000,
        data_dir="data/",
        split_dir="data/",
        md17_version=MD17Version.md17,
        molecule_type=MD17MoleculeType.toluene,
        force_regenerate=False,
        num_timesteps=8,  # Set num_timesteps for replication
    )

    dataloader_dynamic = DataLoader(dataset_dynamic, batch_size=100, shuffle=True)

    # Warm-up iterations
    for _ in range(500):
        next(iter(dataloader_dynamic))

    # Benchmarking with statistics
    times: list[float] = []
    num_batches: int = 10_000

    for rep in tqdm(range(100)):
        start_time = time.time()
        for i, batch in enumerate(dataloader_dynamic):
            if i == num_batches:
                break
        elapsed = time.time() - start_time
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Batch overhead - Mean: {mean_time:.4f} s, Std: {std_time:.4f} s")
    print(f"Latex: \\({mean_time:.3f}{{\\scriptstyle \\pm{std_time:.3f}}}\\)")
