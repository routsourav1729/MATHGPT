#!/usr/bin/env python3
"""
Simple test runner for MATHGPT tests.
Usage: python tests/run_tests.py [options]
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests with specified options"""
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=api", "--cov=src", "--cov=reason"])
    
    # Filter by test type
    if test_type == "api":
        cmd.extend(["-m", "integration"])
    elif test_type == "model":
        cmd.extend(["-m", "unit"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MATHGPT tests")
    parser.add_argument(
        "--type", 
        choices=["all", "api", "model", "fast"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Run with coverage report"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest not installed. Install with: pip install pytest")
        return 1
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)