Below is an updated README with enhanced instructions for macOS OpenMP support using LLVM's Clang:

---

# Decision Tree Classifier

A C++ implementation of a decision tree classifier with Python bindings using [pybind11](https://pybind11.readthedocs.io/en/stable/compiling.html) and [Scikit-build-core](https://github.com/scikit-build/scikit-build-core).

## Usage

```python
$ python3 main.py \
    --n_samples 20000 \
    --n_features 120 \
    --n_classes 4 \
    --random_seed 12 \
    --max_depth 5 \
    --min_samples_split 2 \
    --criterion entropy \
    --min_impurity_decrease 0.001
    --test_size 0.2 
```

```bash
Randomly chosen n_informative: 74
X_train shape: (16000, 120)
X_test shape: (4000, 120)
y_train class distribution: [0.2453125 0.251375  0.2498125 0.2535   ]
y_test class distribution: [0.269  0.2455 0.2475 0.238 ]

Testing with NumPy arrays (zero-copy):
Accuracy: 0.2380
Fit time: 73.4645 seconds
Predict time: 0.0009 seconds

Testing with Python lists (copying):
Accuracy: 0.2380
Fit time: 72.9834 seconds
Predict time: 0.0100 seconds

Performance comparison:
Fit speedup with Eigen: 0.99x
Predict speedup with Eigen: 11.25x
```

## Requirements

- C++ compiler with C++17 support
- CMake 3.15 or higher
- Python 3.11 or higher
- Eigen 3.3 or higher
- OpenMP (optional, for parallelization)

## Installation

### Eigen

**On macOS:**

```bash
brew install eigen
brew info eigen
```

**On Ubuntu/Debian:**

```bash
sudo apt-get install libeigen3-dev
ls /usr/include/eigen3
```

### OpenMP

The Decision Tree Classifier can leverage OpenMP for parallel prediction, improving performance on large datasets. OpenMP is optional but recommended if you plan to work with large data.

#### On Ubuntu/Debian

- **GCC Support:** Most modern GCC versions support OpenMP by default.
  
- **Test OpenMP:**  

  Compile a simple test program to confirm OpenMP support:

  ```bash
  echo "#include <omp.h>
  int main() { return 0; }" > test.c
  gcc -fopenmp test.c -o test && echo "OpenMP is supported"
  ```

- **GCC Version Check:**

  ```bash
  gcc --version
  ```

#### On macOS

- **Default Compiler Note:** The system’s default Clang on macOS does not include OpenMP support.

- **Installation via Homebrew (LLVM):**  

  Install LLVM (which includes an OpenMP-capable Clang) and libomp:

  ```bash
  $ brew install llvm
  $ brew install libomp
  ```
  
- **Environment Configuration:** Since LLVM is keg-only, add LLVM’s bin directory to PATH and point build to the correct headers and libraries. For example, add the following lines to the shell profile (e.g., `~/.zshrc`):

  ```bash
  $ export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
  $ export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
  $ export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
  ```

  Then reload shell or source the file:

  ```bash
  source ~/.zshrc
  ```

- **Test OpenMP with LLVM’s Clang:**  

  Create a test file (`test.c`) with the following content:

  ```c
  #include <omp.h>
  int main() { return 0; }
  ```

  Then compile using LLVM’s Clang with the `-fopenmp=libomp` flag:

  ```bash
  clang -fopenmp=libomp test.c -o test && echo "OpenMP is supported"
  ```
  
  This command tells Clang to use the OpenMP runtime (`libomp`) provided by LLVM.

*Note:* Once OpenMP is installed and detected, the CMake configuration will automatically link the OpenMP libraries when available.

### Option 1: Using Make

The simplest way to install the package is using the provided Makefile:

```bash
make
```

This will:

1. Create a Python virtual environment
2. Install all dependencies
3. Build and install the package

### Option 2: Manual Installation

For manual installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install scikit-build-core pybind11 numpy

pip install -e .
```

## Make Commands

The project includes several useful Makefile commands to streamline development:

| Command       | Description                                                                    |
|---------------|--------------------------------------------------------------------------------|
| `make setup`  | Creates a Python virtual environment and installs dependencies                 |
| `make build`  | Builds the C++ extension module and installs the package                       |
| `make clean`  | Removes build artifacts and temporary files                                    |
| `make rebuild`| Runs `clean` followed by `build` to rebuild from scratch                        |
| `make lint`   | Runs code linting using Ruff                                                   |
| `make run`    | Runs the example script in `main.py`                                           |
| `make all`    | Default command, runs `setup` and `build`                                      |

## Development Workflow

For development, the following workflow is recommended:

1. Make changes to C++ code in `decision_tree_classifier.cpp`
2. Run `make rebuild` to rebuild the extension
3. Test changes with `make run`
4. Format and lint code with `make lint`
