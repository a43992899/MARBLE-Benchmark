from huggingface_hub import snapshot_download
import os

def download_xcodec():
    model_repo = "m-a-p/xcodec_mini_infer"
    target_dir = os.path.join(os.path.dirname(__file__), 'xcodec_mini_infer')
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading {model_repo} to {target_dir}...")
    snapshot_download(
        repo_id=model_repo,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

def download_stage1():
    model_repo = "m-a-p/YuE-s1-7B-anneal-en-cot"
    target_dir = os.path.join(os.path.dirname(__file__), 'stage1')
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading {model_repo} to {target_dir}...")
    snapshot_download(
        repo_id=model_repo,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

if __name__ == "__main__":
    download_xcodec()
    download_stage1()