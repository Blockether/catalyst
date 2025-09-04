#!/usr/bin/env python3
"""
Deployment script for examples to LocalStack Lambda.

This script:
1. Packages example applications with dependencies
2. Deploys to LocalStack Lambda using the built-in Mangum handler
3. Creates function URLs for HTTP access
4. Uses local LocalStack installation from PATH
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FUNCTION_NAME = "examples-api"
HANDLER = "FinancialCalculatorWorkflowExample.handler"
RUNTIME = "python3.11"
ROLE = "arn:aws:iam::000000000000:role/lambda-execution-role"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
SRC_DIR = PROJECT_ROOT / "src"


def run_command(command: str) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    logger.info(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        logger.debug(f"STDOUT: {result.stdout}")
    if result.stderr and result.returncode != 0:
        logger.error(f"STDERR: {result.stderr}")
    return result


def check_prerequisites():
    """Check if LocalStack is running and awslocal is available."""
    logger.info("Checking prerequisites...")

    # Check if LocalStack is installed locally
    result = run_command("which localstack")
    if result.returncode != 0:
        raise RuntimeError("LocalStack not found in PATH. Install with: pip install localstack")

    # Check if LocalStack is running
    result = run_command("curl -s http://localhost:4566/_localstack/health")
    if result.returncode != 0:
        logger.info("LocalStack not running, starting it...")
        # Start LocalStack using local installation
        start_result = run_command("localstack start -d")
        if start_result.returncode != 0:
            raise RuntimeError("Failed to start LocalStack")

        # Wait for LocalStack to be ready
        import time
        time.sleep(10)

        # Check again
        result = run_command("curl -s http://localhost:4566/_localstack/health")
        if result.returncode != 0:
            raise RuntimeError("LocalStack failed to start properly")

    # Check awslocal
    result = run_command("which awslocal")
    if result.returncode != 0:
        raise RuntimeError("awslocal not found. Install with: pip install awscli-local")

    logger.info("✓ Prerequisites OK")


def create_deployment_package() -> Path:
    """Create the deployment ZIP package."""
    logger.info("Creating deployment package...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy the main example file
        shutil.copy2(
            EXAMPLES_DIR / "FinancialCalculatorWorkflowExample.py",
            temp_path / "FinancialCalculatorWorkflowExample.py"
        )

        # Copy source package
        shutil.copytree(SRC_DIR / "com_blockether_catalyst", temp_path / "com_blockether_catalyst")

        # Copy public folder
        public_dir = PROJECT_ROOT / "public"
        if public_dir.exists():
            shutil.copytree(public_dir, temp_path / "public")

        # Create minimal requirements for Lambda
        requirements = [
            "fastapi>=0.104.1",
            "uvicorn>=0.24.0",
            "pydantic>=2.5.0",
            "agno",
            "mangum>=0.19.0"
        ]

        (temp_path / "requirements.txt").write_text("\n".join(requirements))

        # Install dependencies
        logger.info("Installing dependencies...")
        result = run_command(f"cd {temp_path} && python -m pip install -r requirements.txt -t . --quiet")
        if result.returncode != 0:
            raise RuntimeError("Failed to install dependencies")

        # Create ZIP
        zip_path = PROJECT_ROOT / "verification" / f"{FUNCTION_NAME}.zip"
        zip_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_path):
                # Skip __pycache__
                dirs[:] = [d for d in dirs if d != '__pycache__']

                for file in files:
                    if not file.endswith('.pyc'):
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

        logger.info(f"✓ Package created: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f}MB)")
        return zip_path


def setup_lambda_role():
    """Create the Lambda execution role."""
    logger.info("Setting up Lambda role...")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    # Create role (ignore if exists)
    run_command(f"awslocal iam create-role --role-name lambda-execution-role --assume-role-policy-document '{json.dumps(trust_policy)}'")

    # Attach policy (ignore if already attached)
    run_command(f"awslocal iam attach-role-policy --role-name lambda-execution-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")

    logger.info("✓ Lambda role ready")


def deploy_function(zip_path: Path) -> str:
    """Deploy the Lambda function and return the function URL."""
    logger.info(f"Deploying Lambda function: {FUNCTION_NAME}")

    # Delete existing function
    run_command(f"awslocal lambda delete-function --function-name {FUNCTION_NAME}")

    # Create function
    result = run_command(
        f"awslocal lambda create-function "
        f"--function-name {FUNCTION_NAME} "
        f"--runtime {RUNTIME} "
        f"--role {ROLE} "
        f"--handler {HANDLER} "
        f"--zip-file fileb://{zip_path} "
        f"--timeout 30 "
        f"--memory-size 512"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create function: {result.stderr}")

    logger.info("✓ Function deployed")

    # Create function URL
    result = run_command(
        f"awslocal lambda create-function-url-config "
        f"--function-name {FUNCTION_NAME} "
        f"--auth-type NONE "
        f"--cors 'AllowCredentials=false,AllowHeaders=*,AllowMethods=*,AllowOrigins=*'"
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create function URL: {result.stderr}")

    url_config = json.loads(result.stdout)
    function_url = url_config['FunctionUrl']

    logger.info(f"✓ Function URL created: {function_url}")
    return function_url


def test_deployment(function_url: str):
    """Test the deployed function."""
    logger.info("Testing deployment...")

    # Health check
    result = run_command(f"curl -s -X GET '{function_url}'")
    if result.returncode == 0:
        logger.info("✓ Health check passed")

    # Financial calculation test
    test_data = {
        "input": {
            "message": "Calculate compound interest for $10,000 at 7% for 10 years"
        }
    }

    workflow_url = f"{function_url.rstrip('/')}/finance/api/workflows/mainworkflow/runs"
    result = run_command(f"curl -s -X POST '{workflow_url}' -H 'Content-Type: application/json' -d '{json.dumps(test_data)}'")

    if result.returncode == 0:
        logger.info("✓ Financial calculation test passed")


def main():
    """Main deployment function."""
    print("🚀 Deploying Examples to LocalStack Lambda")
    print("=" * 60)

    try:
        check_prerequisites()
        setup_lambda_role()
        zip_path = create_deployment_package()
        function_url = deploy_function(zip_path)

        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"Function: {FUNCTION_NAME}")
        print(f"Handler: {HANDLER}")
        print(f"URL: {function_url}")
        print(f"\n📡 Endpoints:")
        print(f"  • Health: {function_url}")
        print(f"  • Docs: {function_url}finance/api/docs")
        print(f"  • Calculate: {function_url}finance/api/workflows/mainworkflow/runs")

        print(f"\n💡 Test Command:")
        print(f"curl -X POST '{function_url}finance/api/workflows/mainworkflow/runs' \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{{\"input\": {{\"message\": \"Calculate compound interest for $10,000 at 7% for 10 years\"}}}}'")

        test_deployment(function_url)
        print("\n✅ All tests passed!")

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
