#!/usr/bin/env python3
"""
Quick Start Script for Customer Intelligence System
Run this to start both backend and frontend servers
"""

import subprocess
import time
import os
import sys
import signal

def start_backend():
    """Start the FastAPI backend server"""
    print("\n" + "="*60)
    print("Starting Backend Server...")
    print("="*60)
    
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    cmd = [sys.executable, '-m', 'uvicorn', 'server:app', '--reload', '--host', '127.0.0.1', '--port', '8000']
    
    proc = subprocess.Popen(cmd, cwd=backend_dir)
    print("[OK] Backend starting on http://localhost:8000")
    print("[OK] API Docs available at http://localhost:8000/docs")
    return proc

def start_frontend():
    """Start the React frontend server"""
    print("\n" + "="*60)
    print("Starting Frontend Server...")
    print("="*60)
    
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    cmd = ['npm', 'start']
    
    proc = subprocess.Popen(cmd, cwd=frontend_dir, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
    print("[OK] Frontend starting on http://localhost:3000")
    return proc

def main():
    print("\n" + "="*60)
    print("Customer Intelligence System - Quick Start")
    print("="*60)
    
    print("\nChecking dependencies...")
    
    # Check Python packages
    try:
        import fastapi
        import pandas
        import xgboost
        print("[OK] Python dependencies found")
    except ImportError as e:
        print(f"[ERROR] Missing Python package: {e}")
        print("Install requirements: pip install -r backend/requirements.txt")
        sys.exit(1)
    
    # Start servers
    backend_proc = None
    frontend_proc = None
    
    try:
        backend_proc = start_backend()
        time.sleep(5)  # Give backend time to start
        
        frontend_proc = start_frontend()
        time.sleep(10)  # Give frontend time to start
        
        print("\n" + "="*60)
        print("Both servers are running!")
        print("="*60)
        print("\nFrontend:  http://localhost:3000")
        print("Backend:   http://localhost:8000")
        print("API Docs:  http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services...")
        print("="*60 + "\n")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("[OK] All services stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        sys.exit(1)

if __name__ == '__main__':
    main()
