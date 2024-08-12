#!/bin/bash
# shellcheck disable=SC2317

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
CYAN='\033[0;36m'   # Cyan color for check messages
DIM='\033[2m'

# Initialize error counter
ERROR_COUNT=0

# Function to print success messages
print_success() {
    echo -e "✅ ${GREEN}$1${NC}"
}

# Function to print error messages and increment error count
print_error() {
    echo -e "❌ ${RED}$1${NC}"
    ((ERROR_COUNT++))
}

print_warning() {
    echo -e "   ⚠️ ${YELLOW}$1${NC}"
}

print_findings() {
    local message="$1"
    local findings="${2}"
    local icon="${3:-📍}"  # Default to a pin
    if [ -n "$findings" ]; then
        echo -e "   ${icon} ${message}: ${DIM}${findings}${NC}"
    else
        echo -e "   ${icon} ${message}"
    fi
}

print_check() {
    local message="$1"
    local icon="${2:-🕵️}"  # Default to a bullet point if not provided
    echo -e "${CYAN}${icon} ${message}${NC}"
}

prompt_user() {
    local prompt="$1"
    local default="${2:-Y}"
    local REPLY
    local options

    if [[ $default =~ ^[Yy]$ ]]; then
        options="(Y/n)"
    else
        options="(y/N)"
    fi

    while true; do
        read -p "$prompt $options: " -n 1 -r REPLY

        if [ -n "$REPLY" ] ; then
            echo
        fi

        REPLY=${REPLY:-$default}

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0
        elif [[ $REPLY =~ ^[Nn]$ ]]; then
            return 1
        else
            print_error "Invalid input. Please answer Y or N (or press Enter for the default)."
            exit 1
        fi
    done
}

check_git_submodules() {
    print_check "Checking git submodules..." "🧩"

    # Check if we're in a git repository
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        print_error "Not in a git repository"
        return 1
    fi


    # Check if there are any submodules
    if ! submodule_status=$(git submodule status) ; then
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
        current_commit=$(echo "$line" | awk '{print $1}' | sed 's/[^0-9a-f]//g')
        GITSM="git -C $submodule_path"

        case $status_char in
            "-")
                print_error "Submodule not initialized: $submodule_path"
                ((incorrect_submodules++))
                ;;
            "+")
                target_commit=$(git ls-tree --object-only HEAD "$submodule_path")

                if $GITSM merge-base --is-ancestor "$target_commit" "$current_commit" 2> /dev/null; then
                    if [ "$target_commit" = "$current_commit" ]; then
                        print_warning "Submodule has uncommitted changes: $submodule_path"
                    else
                        print_warning "Submodule is ahead of parent repo: $submodule_path"
                    fi
                else
                    print_error "Submodule is behind parent repo: $submodule_path"
                    ((incorrect_submodules++))
                fi
                ;;
            " ")
                print_success "Submodule at correct commit: $submodule_path"
                ;;
            "U")
                print_error "Submodule has merge conflicts: $submodule_path"
                ((incorrect_submodules++))
                ;;
            *)
                print_warning "Unknown status for submodule: $submodule_path"
                ;;
        esac
    done <<< "$submodule_status"

    if [ $incorrect_submodules -eq 0 ]; then
        print_success "All submodules are at the correct commit or ahead"
        return 0
    fi

    print_error "$incorrect_submodules submodule(s) are not at the correct commit"
    echo -e "ℹ️${BLUE}To update all submodules to the correct commit, run:${NC}"
    echo -e "${DIM}   git submodule update --init --recursive${NC}"

    if prompt_user "Would you like to run this command now?"; then
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

    return $incorrect_submodules
}


echo -e "${BLUE}🔍 Checking development environment...${NC}"

check_git_submodules

# Load and run individual check scripts
CHECK_DIR="./kausal_common/development/env-checks"
if [ -d "$CHECK_DIR" ]; then
    for check_script in "$CHECK_DIR"/[0-9][0-9]-*.sh; do
        if [ -f "$check_script" ]; then
            echo -e "${BLUE}🔍 Running check: $(basename "$check_script")${NC}"
            # Source the script, passing necessary functions and variables
            # shellcheck disable=SC1090
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
    echo -e "❌ ${RED}$ERROR_COUNT error(s) detected during the environment check.${NC}"
    exit 1
fi
