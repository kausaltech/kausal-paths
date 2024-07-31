#!/bin/bash

#
# Sync the developer's environment with what is required by the currently active branch.
#


# Colors for output
# Setting the color codes
GREEN='\033[0;32m'  # Green
RED='\033[0;31m'    # Red
BLUE='\033[0;34m'   # Blue
YELLOW='\033[0;33m' # Yellow color for warning messages
NC='\033[0m'        # No color

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
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_git_submodules() {
    echo "🧩 Checking git submodules..."

    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        print_error "Not in a git repository"
        return 1
    fi

    # Check if there are any submodules
    # Check submodule status
    submodule_status=$(git submodule status)
    if [ $? -ne 0 ] ; then
      print_error "'git submodule status' failed"
      return 1
    fi
    if [ -z "$submodule_status" ] ; then
      print_success "No submodules found"
      return 0
    fi
    incorrect_submodules=0

    while IFS= read -r line; do
        status_char=${line:0:1}
        submodule_path=$(echo "$line" | awk '{print $2}')

        case $status_char in
            "-")
                print_error "Submodule not initialized: $submodule_path"
                ((incorrect_submodules++))
                ;;
            "+")
                print_error "Submodule not at the correct commit: $submodule_path"
                ((incorrect_submodules++))
                ;;
            " ")
                print_success "Submodule at correct commit: $submodule_path"
                ;;
            *)
                print_warning "Unknown status for submodule: $submodule_path"
                ;;
        esac
    done <<< "$submodule_status"

    if [ $incorrect_submodules -eq 0 ]; then
        print_success "All submodules are at the correct commit"
    else
        print_error "$incorrect_submodules submodule(s) are not at the correct commit"
        echo -e "${BLUE}ℹ️ To update all submodules to the correct commit, run:${NC}"
        echo -e "${GREEN}   git submodule update --init --recursive${NC}"

        # Prompt user to run the update command
        read -p "Would you like to run this command now? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            echo "Updating submodules..."
            if git submodule update --init --recursive; then
                print_success "Submodules updated successfully"
                incorrect_submodules=0
            else
                print_error "Failed to update submodules"
            fi
        else
            print_warning "Submodules not updated"
        fi

        # Add information about git config submodule.recurse true
        echo -e "${BLUE}ℹ️ To automatically update submodules on git operations, you can set:${NC}"
        echo -e "${GREEN}   git config submodule.recurse true${NC}"
        echo -e "${BLUE}ℹ️ This will ensure submodules are updated whenever you pull or checkout branches.${NC}"
    fi

    return $incorrect_submodules
}

echo -e "${BLUE}🔍 Checking development environment...${NC}"

check_git_submodules

# Load and run individual check scripts
CHECK_DIR="./kausal_common/development/env-checks"
if [ -d "$CHECK_DIR" ]; then
    for check_script in "$CHECK_DIR"/[0-9][0-9]-*.sh; do
        if [ -f "$check_script" ]; then
            echo -e "${BLUE}Running check: $(basename "$check_script")${NC}"
            # Source the script, passing necessary functions and variables
            . "$check_script"
        fi
    done
else
    print_error "Check directory not found: $CHECK_DIR"
fi

echo -e "${BLUE}🏁 Environment check complete.${NC}"

# Print final status and exit
if [ $ERROR_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERROR_COUNT error(s) detected during the environment check.${NC}"
    exit 1
fi
