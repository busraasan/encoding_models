from huggingface_hub import snapshot_download
import os

token = os.environ["HF_API_TOKEN"]

local_dir = snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    use_auth_token=token,
    local_dir="/BRAIN/neuromod-data/static00/apps/hf_cache/llama-2-7b"
    # by default it goes into your HF cache, kk
    # but you can set local_dir="/some/other/path" if you prefer
)
print("Weights downloaded to:", local_dir)
