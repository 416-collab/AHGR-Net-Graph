import os
from mmsdk import mmdatasdk

ROOT = os.path.expanduser("~/GRAPH_SA_DC2M/data/raw")

DATASETS = {
    "mosi": {
        "root": os.path.join(ROOT, "MOSI"),
        "features": {
            "CMU_MOSI_TimestampedWords": mmdatasdk.cmu_mosi.raw["words"],
            "CMU_MOSI_COVAREP": mmdatasdk.cmu_mosi.highlevel["COVAREP"],
            "CMU_MOSI_Opinion_Labels": mmdatasdk.cmu_mosi.labels["Opinion Segment Labels"],
        },
    },
    "mosei": {
        "root": os.path.join(ROOT, "MOSEI"),
        "features": {
            "CMU_MOSEI_TimestampedWords": mmdatasdk.cmu_mosei.raw["words"],
            "CMU_MOSEI_COVAREP": mmdatasdk.cmu_mosei.highlevel["COVAREP"],
            "CMU_MOSEI_Labels": mmdatasdk.cmu_mosei.labels["All Labels"],
        },
    },
}

def download_dataset(name: str):
    cfg = DATASETS[name]
    os.makedirs(cfg["root"], exist_ok=True)
    print(f"\nDownloading {name.upper()} to {cfg['root']}")
    for feature_name, url in cfg["features"].items():
        print(f"  - {feature_name}: {url}")
    dataset = mmdatasdk.mmdataset(cfg["features"], cfg["root"])
    print(f"Finished {name.upper()}")
    print(dataset)

if __name__ == "__main__":
    download_dataset("mosi")
    download_dataset("mosei")
