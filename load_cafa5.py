import os

from datasets import load_dataset
from huggingface_hub import snapshot_download
import tarfile
import concurrent.futures
from tqdm import tqdm



def download_structure_files(
    cache_dir: str,  # Directory to cache the downloaded structures
    num_proc: int = None,  # Number of processes to use for parallel extraction
):
    """Download and extract CAFA5 protein structure files from the Hugging Face hub.
    Args:
        cache_dir (str): Directory to cache the downloaded structures.
        num_proc (int): Number of processes to use for parallel extraction. If None, auto-detects based on CPU count.
    Returns:
        str: Path to the directory containing the extracted structure files.
    """
    num_proc = num_proc or min(4, os.cpu_count() or 1)
    print(f"Using {num_proc} CPU cores for parallel processing")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # Download the CAFA5 Protein Structres
    data_dir = snapshot_download(
        repo_id="wanglab/cafa5",
        local_dir=cache_dir,
        repo_type="dataset",
        allow_patterns="structures_af/*",
        cache_dir=cache_dir,
    )

    structure_dir = os.path.join(data_dir, "structures_af")

    print(f"Downloaded CAFA5 structures to: {structure_dir}")

    # Find all .tar.gz files under structures_af/af_shards/
    tar_file_paths = []
    for subdir in ["af_shards"]:
        shard_dir = os.path.join(structure_dir, subdir)
        if os.path.isdir(shard_dir):
            for fname in os.listdir(shard_dir):
                if fname.endswith(".tar.gz"):
                    tar_file_paths.append(os.path.join(shard_dir, fname))
    print(
        f"Found {len(tar_file_paths)} tar.gz files in {structure_dir}/af_shards"
    )

    extracted_dir = os.path.join(data_dir, "extracted")
    # Ensure extracted directory exists
    os.makedirs(extracted_dir, exist_ok=True)
    print(f"Extracted directory: {extracted_dir}")

    def tar_extract_file(tar_file_path):
        """Helper function to extract a single tar file with flattened structure"""
        try:
            extracted_files = []
            with tarfile.open(tar_file_path, "r:gz") as tar:
                # Get all file members for progress tracking
                members = [member for member in tar.getmembers() if member.isfile()]

                # Extract members but flatten the directory structure
                for member in members:
                    # Extract the file content and write it with the flattened name
                    file_obj = tar.extractfile(member)
                    if file_obj:
                        output_path = os.path.join(
                            extracted_dir, os.path.basename(member.name)
                        )
                        with open(output_path, "wb") as out_file:
                            out_file.write(file_obj.read())
                        extracted_files.append(os.path.basename(member.name))

            return f"Successfully extracted {len(extracted_files)} files from {os.path.basename(tar_file_path)}"
        except Exception as e:
            return f"Failed to extract {os.path.basename(tar_file_path)}: {e}"

    try:
        print(f"Extracting {len(tar_file_paths)} tar files in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_proc, len(tar_file_paths))) as executor:
            results = list(tqdm(
                executor.map(tar_extract_file, tar_file_paths),
                total=len(tar_file_paths),
                desc="Extracting",
            ))

        # Print extraction results
        for result in results:
            print(result)

    except Exception as e:
        print(f"An error occurred during structure tar file extraction: {e}")

    # Print the structure directory
    print(f"Structure files extracted to: {extracted_dir}")

    return extracted_dir


def load_cafa5_dataset(
    dataset: str = "wanglab/cafa5",
    dataset_name: str = "cafa5_reasoning",
    dataset_subset: str = None,
    max_length: int = 2048,
    val_split_ratio: float = 0.1,
    seed: int = 23,
    cache_dir: str = "cafa5_reasoning_cache",
    structure_dir: str = None,
    num_proc: int = None,
    return_as_chat_template: bool = False,
):
    """
    Load CAFA5 dataset, format it into the Protein-LLM format, and split into train/val sets.

    Args:
        dataset_name: Name of the dataset to load
        dataset_subset: Subset of the dataset to load
        max_length: Maximum length for protein sequences
        val_split_ratio: Ratio of training data to use for validation
        seed: Random seed for reproducible splits
        cache_dir: Directory to cache the dataset
        num_proc: Number of CPU cores to use (None = auto-detect)
        return_as_chat_template: Whether to return the dataset as a chat template

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) where test_dataset is the original test split
    """
    try:
        # Auto-detect number of processes if not specified
        num_proc = num_proc or min(4, os.cpu_count() or 1)

        print(f"Using {num_proc} CPU cores for parallel processing")

        # Load CAFA5 dataset
        if dataset_subset:
            dataset = load_dataset(
                dataset,
                name=dataset_name,
                dataset_subset=dataset_subset,
                cache_dir=cache_dir,
            )
        else:
            dataset = load_dataset(dataset, name=dataset_name, cache_dir=cache_dir)

        # Only use the train split from CAFA5
        full_train_dataset = dataset["train"]

        # For testing, limit to 100 datapoints
        # print("Limiting to 100 datapoints for testing...")
        # full_train_dataset = full_train_dataset.select(
        #     range(min(100, len(full_train_dataset)))
        # )

        # Drop rows with null values
        full_train_dataset = full_train_dataset.filter(
            lambda x: x["sequence"] is not None,
            num_proc=num_proc,
        )

        # Truncate protein sequences
        full_train_dataset = full_train_dataset.map(
            lambda x: {
                "sequence": x["sequence"][:max_length],
            },
            num_proc=num_proc,
            desc="Truncating sequences",
        )

        # Set structure paths
        def add_structure_prefix(example):
            if example["structure_path"] is not None:
                example["structure_path"] = os.path.join(
                    structure_dir, example["structure_path"]
                )

            return example

        if structure_dir is None:
            print("No structure directory provided, skipping structure path setting.")
        else:
            print(f"Setting structure paths using directory: {structure_dir}")
            full_train_dataset = full_train_dataset.map(add_structure_prefix)


        # Calculate split sizes
        total_train_size = len(full_train_dataset)
        val_size = int(total_train_size * val_split_ratio)
        train_size = total_train_size - val_size

        # Create train/val split with seed
        train_val_split = full_train_dataset.train_test_split(
            test_size=val_size, seed=seed
        )
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]

        # Use the same validation set as test for now (since we only have train from CAFA5)
        test_dataset = val_dataset

        print(f"CAFA5 Dataset loaded and split successfully:")
        print(f"  - Total original train: {total_train_size} samples")
        print(f"  - Training: {len(train_dataset)} samples ({train_size})")
        print(f"  - Validation: {len(val_dataset)} samples ({val_size})")
        print(f"  - Test: {len(test_dataset)} samples (same as validation)")

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        print(f"Failed to load CAFA5 dataset: {e}")
        print("Returning empty datasets")
        return [], [], []


if __name__ == "__main__":

    # ================================ Cache Directory ================================
    cache_dir = "/Users/arnavshah/Code/DPFunc/cafa5"

    # ================================ Download Structures ================================

    # extracted_structure_dir = download_structure_files(
    #     cache_dir=cache_dir,
    # )
    # For running without downloading structures, you can uncomment the line below
    # extracted_structure_dir = os.path.join(cache_dir, "extracted")

    # ================================ Original Dataset ================================
    train_dataset, val_dataset, test_dataset = load_cafa5_dataset(
        dataset="wanglab/cafa5",
        dataset_name="cafa5_reasoning",
        dataset_subset=None,
        max_length=2048,
        val_split_ratio=0.1,
        seed=23,
        return_as_chat_template=False,
        cache_dir=cache_dir,
        structure_dir='/Users/arnavshah/Code/DPFunc/cafa5/extracted' # '/Users/arnavshah/Code/DPFunc/cafa5/extracted'
    )

    import IPython; IPython.embed()