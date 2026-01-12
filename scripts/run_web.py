#!/usr/bin/env python3
"""
Dr.Case — Запуск Web UI (Streamlit)

Запуск:
    python scripts/run_web.py
    python scripts/run_web.py --port 8501

Примітка:
    Перед запуском Web UI переконайтесь, що API сервер працює:
    python scripts/run_api.py
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Шлях до проекту
project_root = Path(__file__).parent.parent
web_ui_path = project_root / "dr_case" / "web_ui" / "app.py"


def main():
    parser = argparse.ArgumentParser(description='Dr.Case Web UI')
    parser.add_argument('--port', type=int, default=8501, help='Port (default: 8501)')
    parser.add_argument('--host', default='localhost', help='Host (default: localhost)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 Dr.Case — Web UI (Streamlit)")
    print("=" * 60)
    print(f"   App: {web_ui_path}")
    print(f"   URL: http://{args.host}:{args.port}")
    print("=" * 60)
    print()
    print("⚠️  Переконайтесь, що API сервер запущено:")
    print("    python scripts/run_api.py")
    print()
    print("=" * 60)
    
    # Перевіряємо чи є streamlit
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit не встановлено!")
        print("   Встановіть: python -m pip install streamlit")
        sys.exit(1)
    
    # Перевіряємо чи існує файл
    if not web_ui_path.exists():
        print(f"❌ Файл не знайдено: {web_ui_path}")
        sys.exit(1)
    
    print()
    print("🚀 Запуск Streamlit...")
    print()
    
    # Запускаємо Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(web_ui_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser.gatherUsageStats", "false",
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Зупинено")


if __name__ == "__main__":
    main()
