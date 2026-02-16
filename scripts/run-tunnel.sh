#!/bin/bash
set -euo pipefail

# Quick tunnel — generates a random *.trycloudflare.com URL.
# For a custom domain, use a named tunnel + config.yml instead.
cloudflared tunnel --url http://127.0.0.1:8000
