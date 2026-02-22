# Deploying to Hugging Face Spaces

## One-time setup

1. Create a new Space at https://huggingface.co/new-space
   - **Owner:** your HF username
   - **Space name:** `SkinVision`
   - **SDK:** Gradio
   - **Visibility:** Public

2. Add your HF token as a GitHub secret (for automated deploy):
   - Go to GitHub repo → Settings → Secrets → Actions
   - Add secret: `HF_TOKEN` = your Hugging Face write token
     (get it at https://huggingface.co/settings/tokens)

## Manual deploy (one-off)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Push to your Space (replace YOUR_USERNAME)
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='hf_spaces',
    repo_id='YOUR_USERNAME/SkinVision',
    repo_type='space',
    ignore_patterns=['DEPLOY.md'],
)
# Also upload the app and src
api.upload_folder(
    folder_path='.',
    repo_id='YOUR_USERNAME/SkinVision',
    repo_type='space',
    allow_patterns=['app/app.py', 'src/**', 'models/best_model.pth'],
)
"
```

## Automated deploy via GitHub Actions

The `.github/workflows/deploy.yml` workflow automatically pushes to HF Spaces
on every push to `master`, once you've added the `HF_TOKEN` secret and set
`HF_SPACE_ID` in the workflow file to `YOUR_USERNAME/SkinVision`.
