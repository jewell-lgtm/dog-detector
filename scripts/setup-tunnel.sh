#!/bin/bash
set -euo pipefail

# Install cloudflared if missing
command -v cloudflared >/dev/null || brew install cloudflared

# Login (interactive — opens browser)
cloudflared tunnel login

# Create named tunnel
cloudflared tunnel create dog-detector

echo ""
echo "Tunnel created. Next steps:"
echo "  1. Add DNS route:  cloudflared tunnel route dns dog-detector <subdomain>"
echo "  2. Or use quick tunnel:  ./scripts/run-tunnel.sh"
echo ""
echo "=== Camera network isolation ==="
echo "Block the camera's internet access so it can't phone home:"
echo "  - Router firewall: deny WAN for camera's IP (keep LAN port 554 open)"
echo "  - Or: dedicated IoT VLAN/SSID with no WAN access"
echo "  - Mac Mini must still reach camera on port 554 (RTSP)"
