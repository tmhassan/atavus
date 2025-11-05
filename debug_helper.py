#!/usr/bin/env python3
"""
Debug helper to diagnose common issues
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check Python environment and dependencies"""
    print("🔍 Environment Check")
    print("-" * 20)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    # Check required packages
    required_packages = ['pandas', 'numpy', 'sklearn', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def check_file_structure():
    """Check project file structure"""
    print("\n📁 File Structure Check")
    print("-" * 25)
    
    required_files = [
        "src/core/genome_parser.py",
        "src/core/population_analyzer.py",
        "src/utils/validators.py",
        "src/utils/exceptions.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

if __name__ == "__main__":
    env_ok = check_environment()
    files_ok = check_file_structure()
    
    if env_ok and files_ok:
        print("\n🎉 Environment ready for testing!")
    else:
        print("\n⚠️  Please fix the issues above before running tests.")
