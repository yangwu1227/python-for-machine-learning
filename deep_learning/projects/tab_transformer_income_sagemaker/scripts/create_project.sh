#!/bin/bash
set -e

echo "Installing poetry..."
# Install Poetry outside of the environment it manages
curl -sSL https://install.python-poetry.org | python3 -

echo "Adding poetry to PATH..."
POETRY_BIN="$HOME/.local/bin"
export PATH="$POETRY_BIN:$PATH"

# Make path to poetry executable persistent for bash interactive shells
if ! grep -q "$POETRY_BIN" "$HOME/.bashrc" 2>/dev/null; then
  # Suppress duplicate output on the terminal with > /dev/null 
  # Use %s to substitute the Poetry bin path
  printf '\nexport PATH="%s:$PATH"\n' "$POETRY_BIN" | tee -a "$HOME/.bashrc" > /dev/null
  echo "Added Poetry bin path to ~/.bashrc"
fi

# Ensure login shells and sh load ~/.bashrc
for f in "$HOME/.profile" "$HOME/.bash_profile"; do
  # Check if pattern '[ -f ~/.bashrc ] && . ~/.bashrc' does not exist
  if [ ! -f "$f" ] || ! grep -q '\[ -f ~/.bashrc \] && \. ~/.bashrc' "$f" 2>/dev/null; then
    {
      echo ''
      echo '# Load bashrc for login shells or sh'
      echo '[ -f ~/.bashrc ] && . ~/.bashrc'
    } >> "$f"
    echo "Updated $f to source ~/.bashrc"
  fi
done

echo "Initializing conda environment..."
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n tab_tf_env -y python=3.10

echo "Activating conda environment..."
conda activate tab_tf_env

echo "Creating new poetry project..."
poetry new income-classification --python ">=3.10, <3.12" --flat --name src

cd income-classification

echo "Installing dependencies..."
poetry add "polars==0.18.15" "tensorflow==2.13.0" "tensorflow-io==0.32.0" \
           "hydra-core==1.3.2" "boto3==1.26.131" "optuna==3.1.0" \
           "s3fs==2023.6.0" "pymysql==1.1.0"
poetry add "pytest==7.4.2" --group test
poetry add "scikit-learn==1.3.1" "ipykernel==6.25.2" "ipython==8.15.0" \
           "kaleido==0.2.1" "matplotlib==3.8.0" --group notebook

echo "Installing all dependencies..."
poetry install --no-root --all-groups

echo "Project setup complete!"
