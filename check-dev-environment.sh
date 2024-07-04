#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize error counter
ERROR_COUNT=0

# Function to print success messages
print_success() {
    echo -e "${GREEN}✔ $1${NC}"
}

# Function to print error messages and increment error count
print_error() {
    echo -e "${RED}✘ $1${NC}"
    ((ERROR_COUNT++))
}

print_warning() {
    echo -e "${RED}✘ $1${NC}"
}

REQ_FILES="requirements-dev.txt requirements-lint.txt requirements.txt"

check_python_version() {
    echo "📊 Checking Python version requirement..."

    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found"
        return
    fi

    # Extract the Python version requirement from pyproject.toml
    REQUIRED_VERSION=$(grep "requires-python" pyproject.toml | sed -E 's/.*">= ([0-9]+\.[0-9]+)".*/\1/')

    if [ -z "$REQUIRED_VERSION" ]; then
        print_error "Could not find or parse 'requires-python' in pyproject.toml"
        return
    fi

    # Get the current Python version
    CURRENT_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    echo "  🏷️  Required Python version: >=${REQUIRED_VERSION}"
    echo "  🐍 Current Python version: ${CURRENT_VERSION}"

    # Compare versions
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$CURRENT_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        if [ "$REQUIRED_VERSION" = "$CURRENT_VERSION" ]; then
            print_success "Python version matches the requirement"
        else
            print_success "Python version exceeds the minimum requirement"
        fi
    else
        print_error "Python version does not meet the minimum requirement"
        echo -e "${BLUE}ℹ️ You can install the required Python version using:${NC}"
        echo -e "${GREEN}   pyenv install $REQUIRED_VERSION${NC}"
        echo -e "${BLUE}ℹ️ Then, update the Python version in your .envrc file.${NC}"
        echo -e "${BLUE}ℹ️ And reload the environment:${NC}"
        echo -e "${GREEN}   direnv allow${NC}"
}

check_package_versions() {
    echo "📦 Checking installed package versions..."

    # Run pip-sync with dry run
    OUTPUT=$(pip-sync -n $REQ_FILES 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -gt 1 ]; then
        print_error "Error running pip-sync"
        echo "$OUTPUT"
        return 1
    fi

    if echo "$OUTPUT" | grep -q "Would install" || echo "$OUTPUT" | grep -q "Would uninstall"; then
        print_warning "Package version mismatches detected"
        echo "$OUTPUT"

        read -p "Would you like to fix these mismatches? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Fixing package mismatches..."
            pip-sync $REQ_FILES
            if [ $? -eq 0 ]; then
                print_success "Package mismatches resolved"
            else
                print_error "Failed to resolve package mismatches"
                return 1
            fi
        else
            print_error "Package mismatches not resolved"
            return 1
        fi
    else
        print_success "All package versions match requirements"
    fi
}

check_reviewdog() {
    echo "🐶 Checking for reviewdog..."

    # Check if reviewdog exists in ./bin
    if [ -x "./bin/reviewdog" ]; then
        print_success "reviewdog found in ./bin"
        echo "  📍 Path: ./bin/reviewdog"
        return 0
    fi

    # Check if reviewdog exists in PATH
    if command -v reviewdog >/dev/null 2>&1; then
        print_success "reviewdog found in PATH"
        echo "  📍 Path: $(command -v reviewdog)"
        return 0
    fi

    # If we get here, reviewdog was not found
    print_error "reviewdog not found in ./bin or in PATH"

    # Prompt user to install reviewdog, defaulting to yes
    read -p "Would you like to install reviewdog? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_error "reviewdog is required but not installed"
        return 1
    else
        echo "Installing reviewdog..."
        if curl -sfL https://raw.githubusercontent.com/reviewdog/reviewdog/master/install.sh | sh -s; then
            print_success "reviewdog installed successfully"
            echo "  📍 Path: $(command -v reviewdog)"
            return 0
        else
            print_error "Failed to install reviewdog"
            return 1
        fi
    fi
}

check_git_hooks() {
    echo "🔗 Checking git hooks..."

    # Check if pre-commit is available
    if ! command -v pre-commit >/dev/null 2>&1; then
        print_error "pre-commit is not installed"
        return 1
    fi

    # Check if .git directory exists (we're in a git repository)
    if [ ! -d ".git" ]; then
        print_error "Not in a git repository"
        return 1
    fi

    # Check if pre-commit hooks are installed
    if pre-commit install --install-hooks >/dev/null; then
        print_success "Git hooks are properly installed"
    else
        print_error "Error running pre-commit"
        return 1
    fi

    return 0
}

echo -e "${BLUE}🔍 Checking development environment...${NC}"

# Check if running in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_success "🐍 Virtual environment is active"
    echo "  🏠 Virtual environment path: $VIRTUAL_ENV"
else
    print_error "❌ No virtual environment is active"
fi

# Check Python interpreter
PYTHON_PATH=$(which python)
if [ -n "$PYTHON_PATH" ]; then
    print_success "🐍 Python interpreter found"
    echo "  📍 Python path: $PYTHON_PATH"
    echo "  🏷️  Python version: $(python --version)"
else
    print_error "❌ Python interpreter not found in PATH"
    exit 1
fi

check_python_version
check_package_versions
check_reviewdog
check_git_hooks

echo -e "${BLUE}🏁 Environment check complete.${NC}"

# Print final status and exit
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERROR_COUNT error(s) detected during the environment check.${NC}"
    exit 1
fi
