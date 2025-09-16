#!/usr/bin/env python3
"""Simple run script for RAGTube development and testing."""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_server(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI server."""
    cmd = ["uvicorn", "app:app", f"--host={host}", f"--port={port}"]
    if reload:
        cmd.append("--reload")
    
    print(f"Starting RAGTube server on {host}:{port}")
    print("API docs will be available at: http://localhost:8000/docs")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down server...")

def run_tests(coverage=False, verbose=False):
    """Run the test suite."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    print("Running test suite...")
    result = subprocess.run(cmd)
    
    if coverage and result.returncode == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    return result.returncode

def run_eval(evalset=None, comparative=False, **kwargs):
    """Run the evaluation harness."""
    cmd = ["python", "eval/eval.py"]
    
    if evalset:
        cmd.extend(["--evalset", evalset])
    
    if comparative:
        cmd.append("--comparative")
    
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print("Running evaluation...")
    return subprocess.run(cmd).returncode

def create_sample_evalset():
    """Create a sample evaluation set."""
    cmd = ["python", "eval/eval.py", "--create-sample"]
    print("Creating sample evaluation set...")
    return subprocess.run(cmd).returncode

def check_env():
    """Check environment configuration."""
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX"]
    optional_vars = ["COHERE_API_KEY", "YOUTUBE_API_KEY"]
    
    print("Environment Configuration Check")
    print("=" * 40)
    
    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(8, len(value))}...")
        else:
            print(f"❌ {var}: Not set")
            missing_required.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(8, len(value))}... (optional)")
        else:
            print(f"⚠️  {var}: Not set (optional)")
    
    print()
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        print("Please set these in your .env file or environment.")
        return False
    else:
        print("✅ All required environment variables are set!")
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAGTube development helper")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("serve", help="Run the FastAPI server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--evalset", help="Path to evaluation set YAML file")
    eval_parser.add_argument("--comparative", action="store_true", help="Run comparative evaluation")
    eval_parser.add_argument("--alpha", type=float, help="Alpha parameter for hybrid search")
    eval_parser.add_argument("--top-k", type=int, help="Number of top results")
    eval_parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    eval_parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    
    # Sample evalset command
    subparsers.add_parser("create-evalset", help="Create sample evaluation set")
    
    # Environment check command
    subparsers.add_parser("check-env", help="Check environment configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.command == "serve":
        run_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload
        )
        return 0
    
    elif args.command == "test":
        return run_tests(coverage=args.coverage, verbose=args.verbose)
    
    elif args.command == "eval":
        eval_kwargs = {}
        if hasattr(args, 'alpha') and args.alpha is not None:
            eval_kwargs['alpha'] = args.alpha
        if hasattr(args, 'top_k') and args.top_k is not None:
            eval_kwargs['top_k'] = args.top_k
        if hasattr(args, 'no_rerank') and args.no_rerank:
            eval_kwargs['no_rerank'] = True
        if hasattr(args, 'no_compress') and args.no_compress:
            eval_kwargs['no_compress'] = True
        
        return run_eval(
            evalset=args.evalset,
            comparative=args.comparative,
            **eval_kwargs
        )
    
    elif args.command == "create-evalset":
        return create_sample_evalset()
    
    elif args.command == "check-env":
        success = check_env()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
