# Deploying the demo to Hugging Face Spaces

This folder is a ready-to-upload Hugging Face **Space** (Gradio SDK) for the
solar-panel soiling demo. The app code comes from the `solarsoil` package
installed from GitHub; only the model weights need to be added by hand (they are
not in the GitHub repo).

A Space is its own git repo, separate from the GitHub repo. You create it under
your own Hugging Face account — that step can't be automated from here (no HF
token on this machine).

## Files in this bundle (go at the Space root)

| File | Purpose |
|---|---|
| `app.py` | Gradio entry point (copy of `app/app.py`) |
| `requirements.txt` | torch/torchvision (CPU) + `solarsoil` from GitHub |
| `packages.txt` | apt packages OpenCV needs (`libgl1`, `libglib2.0-0`) |
| `README.md` | the Space "card" (title, SDK, version metadata) |

You also upload **two weight files**, which live locally at:

```
artifacts/binary/model.pth      (~90 MB)   -> Space path: artifacts/binary/model.pth
artifacts/severity/model.pth    (~43 MB)   -> Space path: artifacts/severity/model.pth
```

> These weights become publicly downloadable on the Space. That's expected for a
> public demo.

## Option A — web UI (no CLI, easiest)

1. Sign in at https://huggingface.co and click **New** -> **Space**.
2. Owner = your account; Space name e.g. `solar-panel-soiling`; SDK = **Gradio**;
   hardware = **CPU basic (free)**; visibility = **Public**. Create.
3. In the Space's **Files** tab, **Add file -> Upload files** and upload
   `app.py`, `requirements.txt`, `packages.txt`, and `README.md` from this folder.
   (The uploaded `README.md` replaces the auto-generated one — that's correct.)
4. Add the weights at the right paths: **Add file -> Upload files**, then in the
   filename box type `artifacts/binary/model.pth` (creating the folders) and pick
   the local `artifacts/binary/model.pth`. Repeat for
   `artifacts/severity/model.pth`. Files this size go to LFS automatically.
5. The Space rebuilds on each change; watch the **Logs** tab. First build is
   ~5–10 min (it installs torch). When it says *Running*, open the **App** tab.

## Option B — git CLI

```bash
# 1. Create the Space on the website first (as above), then clone it:
git clone https://huggingface.co/spaces/<your-username>/solar-panel-soiling
cd solar-panel-soiling

# 2. Copy the bundle in (from this repo):
cp /Users/senyuzhu/Uni/code/projects/solar_panels/deploy/hf_space/{app.py,requirements.txt,packages.txt,README.md} .

# 3. Add the weights via Git LFS:
git lfs install
git lfs track "*.pth"
mkdir -p artifacts/binary artifacts/severity
cp /Users/senyuzhu/Uni/code/projects/solar_panels/artifacts/binary/model.pth   artifacts/binary/
cp /Users/senyuzhu/Uni/code/projects/solar_panels/artifacts/severity/model.pth artifacts/severity/

git add .gitattributes app.py requirements.txt packages.txt README.md artifacts
git commit -m "Solar panel soiling demo"
git push        # prompts for your HF username + an access token (Settings -> Access Tokens)
```

## After it's live

Copy the Space URL (`https://huggingface.co/spaces/<your-username>/solar-panel-soiling`)
and replace the placeholder in the main repo `README.md` ("Try it live" line).

## Notes

- **The `emoji:` field in `README.md` is required by Hugging Face** for the Space
  thumbnail; it is not shown inside the app. Change it in the Space card / Settings
  if you prefer a different one.
- If the build fails on `torch==2.12.0` / `torchvision==0.27.0` (wheel not found
  for the Space's Python), set `python_version` in `README.md` to `"3.11"` or
  loosen those pins to `torch torchvision`.
- The app degrades gracefully: if a weight file is missing it shows the classical
  soiling view only (no crash), so you can deploy code first and add weights after.
