# FaceCam Puppet (MVP)

## Setup (Windows)
1) Open repo in VS Code
2) Create venv:
   - PowerShell:
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
3) Install deps:
   pip install -r requirements.txt

## Run
python -m facecam.app

## Goal
Preview window first. Then capture in OBS and send via OBS Virtual Camera to TikTok LIVE Studio.
