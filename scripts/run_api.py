#!/usr/bin/env python3
"""
Dr.Case — Запуск API сервера

Запуск:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080
    python scripts/run_api.py --host 127.0.0.1 --port 8000
"""

import sys
import argparse
from pathlib import Path

# Додаємо корінь проекту до path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description='Dr.Case API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏥 Dr.Case — API Server")
    print("=" * 60)
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print("=" * 60)
    
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn не встановлено!")
        print("   Встановіть: pip install uvicorn[standard]")
        sys.exit(1)
    
    try:
        import fastapi
    except ImportError:
        print("❌ fastapi не встановлено!")
        print("   Встановіть: pip install fastapi")
        sys.exit(1)
    
    # Запускаємо сервер
    uvicorn.run(
        "dr_case.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
