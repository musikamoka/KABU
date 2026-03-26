#!/bin/bash
# KRONOS Terminal Launcher (macOS / Linux)

echo ""
echo " ╔══════════════════════════════════════╗"
echo " ║        KRONOS Terminal Launcher      ║"
echo " ╠══════════════════════════════════════╣"
echo " ║  [1] 中文                            ║"
echo " ║  [2] English                         ║"
echo " ║  [3] 日本語                          ║"
echo " ╚══════════════════════════════════════╝"
echo ""
read -p " Select [1/2/3]: " LANG_CHOICE

case $LANG_CHOICE in
  3) LANG=ja ;;
  2) LANG=en ;;
  *) LANG=zh ;;
esac

echo $LANG > "$(dirname "$0")/.lang"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo " [1/4] Activating conda (kronos)..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kronos

echo " [2/4] Starting Kronos API server..."
cd "$SCRIPT_DIR"
python kronos_server.py &
API_PID=$!
sleep 3

echo " [3/4] Starting HTTP server..."
python -m http.server 8080 &
HTTP_PID=$!
sleep 2

echo " [4/4] Opening browser..."
URL="http://localhost:8080/kronos_terminal.html"
if command -v open &>/dev/null; then open "$URL"
elif command -v xdg-open &>/dev/null; then xdg-open "$URL"
fi

echo ""
echo " ✅ KRONOS Terminal is ready!"
echo "    Frontend: $URL"
echo "    Press Ctrl+C to stop all services."
echo ""

trap "kill $API_PID $HTTP_PID 2>/dev/null; echo ' Stopped.'" EXIT
wait
