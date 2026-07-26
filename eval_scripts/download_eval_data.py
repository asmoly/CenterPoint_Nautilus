import requests
from pathlib import Path

def download_data(out_path="v2x_real_lidar64_val.zip"):
    url = "https://ucla.app.box.com/shared/static/est8t7lxirg85ohkgoueietxd0xqpf36"
    out = Path(out_path)

    if out.exists():
        print("ERROR: PATH ALREADY EXISTS")
        return

    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded {out} ({out.stat().st_size / 1e9:.2f} GB)")

download_data()