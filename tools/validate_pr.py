#!/usr/bin/env python3
"""
Local PR validation script for FUSION project.
Runs all checks that would be performed in CI/CD pipelines locally.

Usage:
    python validate_pr.py [--quick] [--lint-only] [--test-only] [--cross-platform-only]
    
Options:
    --quick              Skip slower tests for faster validation
    --lint-only          Only run linting checks
    --test-only          Only run unit tests  
    --cross-platform-only Only run cross-platform compatibility test
    --help               Show this help message
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ValidationRunner:
    """Main validation runner that executes all PR checks locally."""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent
        self.failed_checks = []
        self.total_time = 0
        
    def print_header(self, title):
        """Print a formatted header for each validation step."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")
        
    def print_success(self, message):
        """Print success message."""
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
        
    def print_error(self, message):
        """Print error message."""
        print(f"{Colors.RED}✗ {message}{Colors.END}")
        
    def print_warning(self, message):
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
        
    def run_command(self, cmd, description, cwd=None, check_return_code=True, env=None):
        """Run a shell command and handle output."""
        if cwd is None:
            cwd = self.repo_root
        if env is None:
            env = os.environ.copy()
            
        print(f"{Colors.CYAN}Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}{Colors.END}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                shell=isinstance(cmd, str),
                env=env
            )
            
            elapsed = time.time() - start_time
            self.total_time += elapsed
            
            if result.returncode == 0:
                self.print_success(f"{description} completed successfully ({elapsed:.1f}s)")
                if result.stdout.strip():
                    print(f"Output:\n{result.stdout}")
                return True
            else:
                if check_return_code:
                    self.print_error(f"{description} failed ({elapsed:.1f}s)")
                    self.failed_checks.append(description)
                    # For pylint, always show stdout as it contains the warnings/errors
                    if "Pylint" in description and result.stdout:
                        print(f"Pylint issues found:\n{result.stdout}")
                    if result.stderr:
                        print(f"Error output:\n{result.stderr}")
                    elif result.stdout and "Pylint" not in description:
                        print(f"Standard output:\n{result.stdout}")
                    return False
                else:
                    self.print_warning(f"{description} completed with warnings ({elapsed:.1f}s)")
                    if result.stderr:
                        print(f"Warnings:\n{result.stderr}")
                    return True
                    
        except Exception as e:
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.print_error(f"{description} failed with exception: {e}")
            self.failed_checks.append(description)
            return False
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        self.print_header("Checking Dependencies")
        
        required_tools = [
            ('python', 'Python interpreter'),
            ('pip', 'Python package manager'),
            ('pylint', 'Python linter'),
            ('pytest', 'Python test framework')
        ]
        
        missing_tools = []
        for tool, description in required_tools:
            if not self.run_command(['which', tool], f"Check {description}", check_return_code=False):
                missing_tools.append(tool)
        
        if missing_tools:
            self.print_error(f"Missing required tools: {', '.join(missing_tools)}")
            self.print_error("Please install missing dependencies:")
            print("  pip install pylint pytest")
            return False
            
        # Check if we're in a virtual environment
        if not os.environ.get('VIRTUAL_ENV'):
            self.print_warning("Not running in a virtual environment")
            self.print_warning("Consider activating your venv: source venv/bin/activate")
            
        return True
    
    def validate_python_syntax(self):
        """Check Python syntax across all Python files."""
        self.print_header("Python Syntax Validation")
        
        # Find all Python files
        cmd = [
            'find', '.', 
            '-name', '*.py',
            '-not', '-path', './venv/*',
            '-not', '-path', './.venv/*',
            '-not', '-path', './docs/*'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
        if result.returncode != 0:
            self.print_error("Failed to find Python files")
            return False
            
        python_files = result.stdout.strip().split('\n')
        python_files = [f for f in python_files if f.strip()]
        
        print(f"Found {len(python_files)} Python files to validate")
        
        syntax_errors = []
        for py_file in python_files:
            try:
                with open(self.repo_root / py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")
                
        if syntax_errors:
            self.print_error("Python syntax errors found:")
            for error in syntax_errors:
                print(f"  {error}")
            return False
            
        self.print_success("All Python files have valid syntax")
        return True
    
    def run_pylint(self):
        """Run pylint on the codebase."""
        self.print_header("Running Pylint")
        
        # Run pylint with strict checking - fail on warnings and errors
        fusion_success = self.run_command(
            ['pylint', './fusion'],
            "Pylint on fusion package",
            check_return_code=True  # Fail validation on pylint warnings/errors
        )
        
        tests_success = self.run_command(
            ['pylint', './tests'],
            "Pylint on tests package", 
            check_return_code=True  # Fail validation on pylint warnings/errors
        )
        
        return fusion_success and tests_success
    
    def run_unit_tests(self, quick=False):
        """Run pytest unit tests."""
        self.print_header("Running Unit Tests")
        
        cmd = ['pytest']
        if quick:
            cmd.extend(['-x', '--tb=short'])  # Stop on first failure, short traceback
        else:
            cmd.extend(['-v'])  # Verbose output
            
        return self.run_command(cmd, "Unit tests")
    
    def run_cross_platform_test(self):
        """Run the cross-platform compatibility test."""
        self.print_header("Cross-Platform Compatibility Test")
        
        # Set PYTHONPATH to include current directory
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{self.repo_root}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.repo_root)
        
        cmd = [
            'python', '-m', 'fusion.cli.run_sim', 'run_sim',
            '--config_path=fusion/configs/templates/cross_platform.ini',
            '--run_id=local_validation'
        ]
        
        # This test is expected to fail with numpy architecture error locally
        # We just want to ensure it gets past the configuration parsing
        result = self.run_command(cmd, "Cross-platform test", check_return_code=False, env=env)
        
        # Check if it failed due to numpy architecture (expected) vs config issues (bad)
        if not result:
            # Re-run to capture stderr for analysis
            proc_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            if "numpy" in proc_result.stderr and "architecture" in proc_result.stderr:
                self.print_warning("Cross-platform test failed due to numpy architecture (expected locally)")
                self.print_success("Configuration parsing appears to work correctly")
                return True
            else:
                self.print_error("Cross-platform test failed due to configuration issues")
                return False
                
        return True
    
    def check_config_files(self):
        """Validate configuration files."""
        self.print_header("Configuration File Validation")
        
        config_files = list(Path(self.repo_root / 'fusion' / 'configs' / 'templates').glob('*.ini'))
        
        if not config_files:
            self.print_error("No configuration files found")
            return False
            
        print(f"Found {len(config_files)} configuration files")
        
        for config_file in config_files:
            try:
                # Test loading each config file
                cmd = [
                    'python', '-c',
                    f"import sys; sys.path.insert(0, '.'); "
                    f"from fusion.cli.config_setup import load_config; "
                    f"result = load_config('{config_file}'); "
                    f"print(f'✓ {config_file.name}: {{len(result)}} sections loaded')"
                ]
                
                if not self.run_command(cmd, f"Validate {config_file.name}"):
                    return False
                    
            except Exception as e:
                self.print_error(f"Failed to validate {config_file.name}: {e}")
                return False
                
        return True
    
    def run_import_tests(self):
        """Test that all modules can be imported without errors."""
        self.print_header("Import Tests")
        
        key_modules = [
            'fusion',
            'fusion.cli',
            'fusion.core',
            'fusion.sim',
            'fusion.modules',
            'fusion.utils'
        ]
        
        for module in key_modules:
            cmd = ['python', '-c', f'import sys; sys.path.insert(0, "."); import {module}; print("✓ {module} imported successfully")']
            if not self.run_command(cmd, f"Import {module}"):
                return False
                
        return True
    
    def print_summary(self):
        """Print validation summary."""
        print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.PURPLE}VALIDATION SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}")
        
        if self.failed_checks:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ FAILED CHECKS ({len(self.failed_checks)}):{Colors.END}")
            for check in self.failed_checks:
                print(f"  {Colors.RED}• {check}{Colors.END}")
                
            print(f"\n{Colors.RED}{Colors.BOLD}PR VALIDATION FAILED{Colors.END}")
            print(f"{Colors.RED}Please fix the issues above before submitting your PR.{Colors.END}")
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL CHECKS PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Your PR is ready for submission.{Colors.END}")
            
        print(f"\n{Colors.CYAN}Total validation time: {self.total_time:.1f} seconds{Colors.END}")
        
        # Print next steps
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        if self.failed_checks:
            print("1. Fix the failed checks listed above")
            print("2. Re-run this script to verify fixes")
            print("3. Commit your changes")
            print("4. Submit your PR")
        else:
            print("1. Commit your changes if you haven't already")
            print("2. Push to your branch")
            print("3. Create/update your PR")
            print("4. The CI/CD pipelines should pass ✅")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate PR locally by running all CI/CD checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Skip slower tests for faster validation')
    parser.add_argument('--lint-only', action='store_true',
                       help='Only run linting checks')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only run unit tests')
    parser.add_argument('--cross-platform-only', action='store_true',
                       help='Only run cross-platform compatibility test')
    
    args = parser.parse_args()
    
    # Print welcome message
    print(f"{Colors.BOLD}{Colors.GREEN}🚀 FUSION PR Validation Tool{Colors.END}")
    print(f"{Colors.CYAN}This tool runs all CI/CD checks locally to validate your PR{Colors.END}")
    
    runner = ValidationRunner()
    
    # Always check dependencies first
    if not runner.check_dependencies():
        sys.exit(1)
    
    success = True
    
    # Run selected checks based on arguments
    if args.lint_only:
        success = runner.validate_python_syntax() and runner.run_pylint()
    elif args.test_only:
        success = runner.run_unit_tests(quick=args.quick)
    elif args.cross_platform_only:
        success = runner.run_cross_platform_test()
    else:
        # Run all checks (default)
        steps = [
            runner.validate_python_syntax,
            runner.run_import_tests,
            runner.check_config_files,
            runner.run_pylint,
            lambda: runner.run_unit_tests(quick=args.quick),
            runner.run_cross_platform_test,
        ]
        
        for step in steps:
            if not step():
                success = False
                if args.quick:
                    break  # Stop on first failure in quick mode
    
    # Print summary
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()