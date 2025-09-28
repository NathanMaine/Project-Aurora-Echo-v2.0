#!/usr/bin/env python3
"""
Project Aurora Echo - Testing Suite
Comprehensive testing script for code quality and basic functionality validation.
"""

import ast
import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def check_syntax(filepath):
    """Check Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def run_syntax_tests():
    """Run comprehensive syntax validation."""
    print_header("SYNTAX VALIDATION TESTS")

    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip common directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', '.git']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    total_files = len(python_files)
    valid_files = 0
    errors = []

    print(f"Scanning {total_files} Python files...")

    for filepath in python_files:
        is_valid, error = check_syntax(filepath)
        if is_valid:
            valid_files += 1
        else:
            errors.append(f"{filepath}: {error}")

    print(f"\n📊 Results:")
    print(f"   Total files: {total_files}")
    print(f"   Valid files: {valid_files}")
    print(f"   Error files: {len(errors)}")

    if errors:
        print(f"\n❌ Syntax errors found in {len(errors)} files:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        return False
    else:
        print("\n✅ All Python files have valid syntax!")
        return True

def run_import_tests():
    """Test basic module imports."""
    print_header("IMPORT STRUCTURE TESTS")

    tests = [
        ("Core app structure", "import app"),
        ("Services module", "import services"),
        ("LLM service", "from services import llm_service"),
        ("ASR service", "from services import asr_service"),
        ("Orchestrator", "from services import orchestrator"),
    ]

    passed = 0
    total = len(tests)

    for test_name, import_stmt in tests:
        try:
            # Use subprocess to avoid import pollution
            result = subprocess.run(
                [sys.executable, '-c', import_stmt],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
                print(f"   Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_name}: TIMEOUT")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n📊 Import Tests: {passed}/{total} passed")
    return passed == total

def run_fastapi_tests():
    """Test FastAPI framework setup."""
    print_header("FRAMEWORK VALIDATION TESTS")

    try:
        # Test FastAPI import and basic setup
        result = subprocess.run([
            sys.executable, '-c', '''
import sys
try:
    from fastapi import FastAPI
    app = FastAPI(title="Test Aurora Echo", description="Testing setup")
    print("FastAPI setup successful")
except ImportError as e:
    print(f"FastAPI not available: {e}")
    sys.exit(1)
'''
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ FastAPI framework: AVAILABLE")
            return True
        else:
            print("❌ FastAPI framework: UNAVAILABLE")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ FastAPI test error: {e}")
        return False

def check_dependencies():
    """Check for required dependencies."""
    print_header("DEPENDENCY CHECK")

    required_packages = [
        'fastapi', 'uvicorn', 'websockets', 'pyaudio', 'numpy',
        'torch', 'python-dotenv', 'requests', 'pyttsx3'
    ]

    available = 0
    total = len(required_packages)

    for package in required_packages:
        try:
            result = subprocess.run([
                sys.executable, '-c', f'import {package}'
            ], capture_output=True, timeout=5)

            if result.returncode == 0:
                print(f"✅ {package}: INSTALLED")
                available += 1
            else:
                print(f"❌ {package}: MISSING")
        except subprocess.TimeoutExpired:
            print(f"⏰ {package}: TIMEOUT")
        except Exception as e:
            print(f"❌ {package}: ERROR - {e}")

    print(f"\n📊 Dependencies: {available}/{total} available")

    if available < total:
        print("\n💡 Tip: Run 'pip install -r requirements.txt' to install missing dependencies")

    return available == total

def main():
    """Run all tests."""
    print("🚀 Project Aurora Echo - Testing Suite")
    print("=====================================")

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    results = []

    # Run all test suites
    results.append(("Syntax Validation", run_syntax_tests()))
    results.append(("Import Tests", run_import_tests()))
    results.append(("Framework Tests", run_fastapi_tests()))
    results.append(("Dependencies", check_dependencies()))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\n📊 Overall: {passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 All tests passed! Your Project Aurora Echo setup is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suites failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())