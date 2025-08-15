#!/bin/bash
# Simple shell script wrapper for PR validation
# Usage: ./validate_pr.sh [quick|lint|test|cross-platform]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located and go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}🚀 FUSION PR Validation${NC}"
echo "=========================="

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not detected${NC}"
    echo -e "${YELLOW}   Consider running: source venv/bin/activate${NC}"
    echo ""
fi

# Check if Python validation script exists
if [[ ! -f "tools/validate_pr.py" ]]; then
    echo -e "${RED}❌ tools/validate_pr.py not found${NC}"
    echo "Please ensure you're in the FUSION project root directory"
    exit 1
fi

# Run validation based on argument
case "${1:-full}" in
    "quick")
        echo -e "${BLUE}Running quick validation...${NC}"
        python tools/validate_pr.py --quick
        ;;
    "lint")
        echo -e "${BLUE}Running linting checks only...${NC}"
        python tools/validate_pr.py --lint-only
        ;;
    "test")
        echo -e "${BLUE}Running unit tests only...${NC}"
        python tools/validate_pr.py --test-only
        ;;
    "cross-platform")
        echo -e "${BLUE}Running cross-platform test only...${NC}"
        python tools/validate_pr.py --cross-platform-only
        ;;
    "full"|"")
        echo -e "${BLUE}Running complete validation...${NC}"
        python tools/validate_pr.py
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [quick|lint|test|cross-platform|full]"
        echo ""
        echo "Options:"
        echo "  quick           - Quick validation (stops on first failure)"
        echo "  lint            - Only run linting checks"
        echo "  test            - Only run unit tests"
        echo "  cross-platform  - Only run cross-platform compatibility test"
        echo "  full            - Complete validation (default)"
        echo "  help            - Show this help message"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use '$0 help' to see available options"
        exit 1
        ;;
esac