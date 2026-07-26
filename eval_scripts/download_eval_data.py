import zipfile
import requests
from pathlib import Path

def download_data():
    url = "https://ucla.app.box.com/shared/static/est8t7lxirg85ohkgoueietxd0xqpf36"

    zip_path = Path("v2x_real_lidar64_val.zip")
    out_dir = Path("v2x_real_lidar64_val")

    if zip_path.exists():
        print("ERROR: DOWNLOADED ZIP ALREADY EXISTS")
    else:
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    if out_dir.exists():
        print("ERROR: EXTRACTED FILES ALREADY EXIST")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        print(f"Extracted to {out_dir}")

    print("Finished downloading and extracting")

download_data()