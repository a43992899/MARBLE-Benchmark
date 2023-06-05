import json
import os
import pathlib
import urllib.request
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='./data/GS')
    config = parser.parse_args()

    # Parse asset specification JSON
    _ASSET_PATHS = set()
    json_path = "benchmark/tasks/GS/assets/giantsteps.json"
    with open(json_path, "r") as f:
        d = json.load(f)

    for tag, asset in tqdm.tqdm(d.items()):
        if tag != tag.upper():
            raise AssertionError("Tags should be uppercase")
        if "checksum" not in asset:
            raise AssertionError("Missing checksum")
        try:
            asset["path_rel"] = pathlib.PurePosixPath(
                asset["path_rel"].replace("datasets/giantsteps/", ""))
        except BaseException:
            print(asset["path_rel"])
            raise AssertionError("Invalid path")
        asset["path_rel"] = os.path.join(config.data_path, asset["path_rel"])
        if asset["path_rel"] in _ASSET_PATHS:
            raise AssertionError("Duplicate path")
        _ASSET_PATHS.add(asset["path_rel"])

        if not pathlib.Path(asset["path_rel"]).is_file():
            new_path = str(asset["path_rel"]).split("/")
            new_path = "/".join(new_path[:-1])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            # check if file exists
            if not os.path.isfile(asset["path_rel"]):
                data = urllib.request.urlretrieve(asset["url"], asset["path_rel"])
