"""
python tools/generate_fineweb_subset.py --num_samples=500_000
"""
import datasets
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=500_000,
        help="number of samples to sample from the FineWeb.",
    )

def main(argv):
    num_samples = FLAGS.num_samples
    fw_data = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        # streaming=True
    )
    subset = fw_data.select(np.arange(num_samples))
    num_k = num_samples // 1000
    save_path = f"dataset_cache/fw_subset_{num_k}k"
    subset.save_to_disk(save_path, num_shards=10)

if __name__ == "__main__":
    set_args()
    app.run(main)





