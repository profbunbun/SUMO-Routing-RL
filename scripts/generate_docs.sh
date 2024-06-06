#!/bin/bash

# Base directories
PROJECT_ROOT="$(dirname "$(dirname "$0")")"
DOCS_DIR="$PROJECT_ROOT/docs/source"
SRC_DIR="$PROJECT_ROOT/src"

# Create docs directory if it doesn't exist
mkdir -p "$DOCS_DIR"

# Create root index.rst
ROOT_INDEX_FILE="$DOCS_DIR/index.rst"
if [ ! -f "$ROOT_INDEX_FILE" ]; then
    cat > "$ROOT_INDEX_FILE" <<EOL
.. Your Project Name documentation master file, created by
   sphinx-quickstart on Sun Jun  4 2024.

Welcome to Your Project Name's documentation!
=============================================

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Module Index
============

.. toctree::
   :maxdepth: 2

   config
   environment/index
   rl_algorithms/index
   utils

Indices and tables
==================

* :ref:\`genindex\`
* :ref:\`modindex\`
* :ref:\`search\`
EOL
    echo "Created $ROOT_INDEX_FILE"
else
    echo "$ROOT_INDEX_FILE already exists"
fi

# Function to create index.rst for directories
create_dir_index() {
    local dir_path="$1"
    local index_file="$dir_path/index.rst"
    shift
    local modules=("$@")

    mkdir -p "$dir_path"

    if [ ! -f "$index_file" ]; then
        cat > "$index_file" <<EOL
$(basename $dir_path) Module
==================

.. toctree::
   :maxdepth: 2

EOL
        for module in "${modules[@]}"; do
            echo "   $module" >> "$index_file"
        done
        echo "Created $index_file"
    else
        echo "$index_file already exists"
    fi
}

# Function to create module rst files
create_module_rst() {
    local module_path="$1"
    local module_file="$module_path.rst"
    local module_name="${module_path//\//.}"

    mkdir -p "$(dirname "$module_file")"

    if [ ! -f "$module_file" ]; then
        cat > "$module_file" <<EOL
$module_name Module
$(printf '=%.0s' {1..${#module_name}})
.. automodule:: ${module_name#docs.source.}
    :members:
    :undoc-members:
    :show-inheritance:
EOL
        echo "Created $module_file"
    else
        echo "$module_file already exists"
    fi
}

# Create directory indices and module rst files

create_dir_index "$DOCS_DIR/config" "config"
create_module_rst "$DOCS_DIR/config/config"

create_dir_index "$DOCS_DIR/environment" \
    "bus_stop" "env" "observation" "outmask" "person" "person_manager" \
    "reward" "ride_select" "sim_manager" "vehicle" "vehicle_manager"

for module in bus_stop env observation outmask person person_manager reward \
              ride_select sim_manager vehicle vehicle_manager; do
    create_module_rst "$DOCS_DIR/environment/$module"
done

create_dir_index "$DOCS_DIR/rl_algorithms" "dqn/index" "exploration" "memory_buffers" "models" "ppo"
create_dir_index "$DOCS_DIR/rl_algorithms/dqn" "agent" "peragent"

for module in agent peragent; do
    create_module_rst "$DOCS_DIR/rl_algorithms/dqn/$module"
done

for module in exploration memory_buffers models ppo; do
    create_module_rst "$DOCS_DIR/rl_algorithms/$module"
done

create_dir_index "$DOCS_DIR/utils" "connect" "net_parser" "utils"

for module in connect net_parser utils; do
    create_module_rst "$DOCS_DIR/utils/$module"
done
