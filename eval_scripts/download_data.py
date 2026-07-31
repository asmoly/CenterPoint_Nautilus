"""
This script downloads the v2x validation data, specifically the v2x_real_lidar64_val data
It downloads it from this site: https://mobility-lab.seas.ucla.edu/v2x-real/

How to use:
You can specify which dataset you want to download using the arguents test, val, and train

python download_data.py test
python download_data.py val
python download_data.py train

All datasets will be downloaded to the outdir folder "v2x_real_lidar64"
"""

import zipfile
import requests
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["test", "val", "train"],
        help="Which dataset to download",
    )
    return parser.parse_args()

def download_data():
    args = parse_args()

    print(f"Downloading {args.dataset} dataset")

    urls_to_download = []
    if args.dataset == "test":
        urls_to_download.append(("test.zip", "https://ucla.app.box.com/shared/static/429ak8yk8pawnd84xqx5opcsl0p2mvrp"))
    elif args.dataset == "val":
        urls_to_download.append(("val.zip", "https://ucla.app.box.com/shared/static/est8t7lxirg85ohkgoueietxd0xqpf36"))
    elif args.dataset == "train":
        urls_to_download.append(("train1.zip", "https://ucla.app.box.com/shared/static/nlnnfbx52m0rjdjwalckcnj3xoc6kckq"))
        urls_to_download.append(("train2.zip", "https://ucla.app.box.com/shared/static/98rgies3hn0uw6d5oecajsru99lmvi0q"))
        urls_to_download.append(("train3.zip", "https://ucla.app.box.com/shared/static/d1fuuvx4hisfxkxi1h0gipxfn7tpaony"))
        urls_to_download.append(("train4.zip", "https://ucla.app.box.com/shared/static/tre8f2n816n4dfqodp74cnqfz7on5qw2"))

    out_dir = Path("v2x_real_lidar64")
    out_dir.mkdir(parents=True, exist_ok=True)

    for zip_path, url in urls_to_download:
        zip_path = Path(zip_path)
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        print(f"Extracted {zip_path} to {out_dir}")

    print("Finished downloading and extracting")

download_data()