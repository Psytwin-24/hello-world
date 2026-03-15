#!/usr/bin/env bash
# ============================================================
# VM Setup Script for NSE/BSE Options Trading Bot
# Tested on Ubuntu 22.04 LTS
#
# Run as: bash scripts/setup_vm.sh
# ============================================================

set -euo pipefail

echo "=========================================="
echo " NSE/BSE Options Bot — VM Setup"
echo "=========================================="

# --- System packages ---
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential libssl-dev libffi-dev \
    postgresql postgresql-contrib \
    redis-server \
    ta-lib \
    git curl wget htop tmux \
    libpq-dev

# --- TA-Lib (required by ta-lib Python package) ---
if ! pkg-config --exists ta_lib 2>/dev/null; then
    echo "Installing TA-Lib from source..."
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    sudo ldconfig
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
fi

# --- Python virtual environment ---
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

echo "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# --- PostgreSQL setup ---
sudo systemctl start postgresql
sudo systemctl enable postgresql

sudo -u postgres psql << 'EOF'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trader') THEN
    CREATE USER trader WITH PASSWORD 'trader';
  END IF;
END
$$;

SELECT 'CREATE DATABASE options_bot OWNER trader'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'options_bot')\gexec
GRANT ALL PRIVILEGES ON DATABASE options_bot TO trader;
EOF

echo "PostgreSQL setup complete."

# --- Redis ---
sudo systemctl start redis-server
sudo systemctl enable redis-server
echo "Redis started."

# --- Environment file ---
if [ ! -f .env ]; then
    cp config/.env.example .env
    echo ""
    echo "⚠️  .env file created from template."
    echo "    Edit .env and fill in your broker credentials before running live."
fi

# --- Systemd service for the bot ---
BOT_DIR=$(pwd)
VENV_DIR="$BOT_DIR/venv"
USER_NAME=$(whoami)

sudo tee /etc/systemd/system/options-bot.service > /dev/null << EOF
[Unit]
Description=NSE/BSE Options Trading Bot
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$BOT_DIR
ExecStart=$VENV_DIR/bin/python src/bot.py --paper
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=$BOT_DIR

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/options-dashboard.service > /dev/null << EOF
[Unit]
Description=NSE/BSE Options Bot Dashboard
After=options-bot.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$BOT_DIR
ExecStart=$VENV_DIR/bin/python src/bot.py --dashboard
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=$BOT_DIR

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo ""
echo "Systemd services created:"
echo "  options-bot.service       — trading bot"
echo "  options-dashboard.service — web dashboard (port 8050)"
echo ""
echo "To start: sudo systemctl start options-bot options-dashboard"
echo "To enable on boot: sudo systemctl enable options-bot options-dashboard"
echo "To view logs: sudo journalctl -fu options-bot"

# --- Cron for auto-research (backup if systemd not used) ---
CRON_CMD="$VENV_DIR/bin/python $BOT_DIR/src/bot.py --research >> $BOT_DIR/logs/research.log 2>&1"
(crontab -l 2>/dev/null; echo "0 */4 * * 1-5 $CRON_CMD") | crontab -
echo "Cron job added: auto-research every 4h on weekdays"

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your broker credentials"
echo "  2. Set PAPER_TRADING=true for paper mode"
echo "  3. Run: python src/bot.py --paper"
echo "  4. Dashboard: http://your-vm-ip:8050"
echo ""
echo "IMPORTANT: Always start with paper trading and backtesting."
echo "Only switch to live after thorough validation."
