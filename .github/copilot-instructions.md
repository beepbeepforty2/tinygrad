# tinygrad AI Coding Agent Instructions

## Philosophy

tinygrad is a **tensor library** focused on beauty and minimalism, matching PyTorch/JAX functionality with minimal code.

**Core principles:**
- Every line must earn its keep - prefer readability over cleverness
- 10 well-designed lines can have the impact of 1000
- **Never mix functionality changes with whitespace changes**
- All functionality changes **must be tested**

## Code Style

- **2-space indentation** (not 4, not tabs)
- Maximum **150 characters per line**
- Match existing style - consistency is critical
- Follow `ruff.toml` and `mypy.ini` configurations
- Install pre-commit hooks: `pre-commit install`

## Architecture Overview

tinygrad uses a **lazy execution model** with a multi-stage compilation pipeline:

1. **Tensor graph** (`tinygrad/tensor.py`) - High-level operations (user-facing API)
2. **UOp graph** (`tinygrad/uop/`) - Unified operation IR (replaces old LazyBuffer)
3. **Schedule** (`tinygrad/schedule/`, `tinygrad/engine/schedule.py`) - Kernelize and create execution plan
4. **Kernel optimization** (`tinygrad/codegen/opt/`) - Apply hardware-specific optimizations
5. **Code generation** (`tinygrad/renderer/`) - Generate device code (PTX, Metal, OpenCL, etc.)
6. **Runtime** (`tinygrad/runtime/ops_*.py`) - Device-specific execution

**Key concepts:**
- **UOp**: Unified operation representation - the IR that replaced LazyBuffer
- **Kernelize**: Convert Tensor graph → UOp graph with KERNEL nodes
- **Schedule**: Plan execution order, manage dependencies via AFTER nodes
- **Realize**: Execute scheduled kernels, allocate buffers
- **Lazy evaluation**: Operations build graphs until `.realize()` or `.numpy()` is called

## Critical Development Workflows

### Running Tests

```bash
# Install test dependencies
python3 -m pip install -e '.[testing]'

# Run specific test file
python3 test/test_ops.py

# Run full suite
python3 -m pytest test/

# Run with specific backend
CUDA=1 python3 test/test_ops.py
```

### Debugging with DEBUG Levels

Environment variable `DEBUG=[1-7]` controls output verbosity:

- `DEBUG=1`: List devices being used
- `DEBUG=2`: Performance metrics (timing, memory, bandwidth per kernel)
- `DEBUG=3`: Buffers (shape, dtype, strides) and kernel optimizations
- `DEBUG=4`: Generated kernel code
- `DEBUG=5`: UOp AST (intermediate representation)
- `DEBUG=6`: Linearized UOps
- `DEBUG=7`: Generated assembly

Example: `DEBUG=4 python3 -c "from tinygrad import Tensor; Tensor.ones(10,10).sum().numpy()"`

### Process Replay (Testing Kernel Changes)

PRs marked with `[pr]` in title trigger process replay - compares generated kernels against master.

**Critical for refactors/speedups** that shouldn't change behavior.

```bash
# Capture kernels in your branch
CAPTURE_PROCESS_REPLAY=1 python3 test/test_ops.py

# Switch to master and compare
git checkout master
python3 test/external/process_replay/process_replay.py
```

## Backend/Device System

- Each backend in `tinygrad/runtime/ops_*.py` (e.g., `ops_cuda.py`, `ops_metal.py`)
- Backends need ~25 low-level ops to work
- Select device: `CUDA=1`, `METAL=1`, `CPU=1`, etc.
- Check default: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`
- HCQ (Hardware Command Queue) backends: NV, AMD, QCOM - lower-level GPU control

## Common Patterns

### Testing Pattern

```python
import numpy as np
from tinygrad import Tensor

# Compare against numpy/torch
t = Tensor([1,2,3])
result = (t * 2 + 1).numpy()
np.testing.assert_allclose(result, np.array([3,5,7]))
```

### Schedule Inspection

```python
from tinygrad import Tensor
from tinygrad.engine.schedule import create_schedule_with_vars

a = Tensor.ones(10, 10)
b = (a + 1).sum()
schedule, var_vals = b.schedule_with_vars()
print(f"Generated {len(schedule)} kernels")
```

### Working with UOps

```python
from tinygrad import Tensor
from tinygrad.uop.ops import Ops

t = Tensor.ones(10)
# Access underlying UOp
print(t.uop.op)  # Will show the op type
print(t.uop.toposort())  # Get all UOps in dependency order
```

## PR Guidelines

**Will be closed:**
- Code golf or deleting `\n`s without reducing complexity
- Docs/whitespace changes from non-core contributors
- "Speedup" claims without benchmarks
- Complex PRs without breaking into smaller chunks
- Changes to `extra/` unless code is broken

**Welcome:**
- Bug fixes with regression tests
- Bounty solutions (see spreadsheet)
- Features with tests (3-line features > 300-line features)
- Clear-win refactors that pass process replay
- Tests, fuzzers, dead code removal from `tinygrad/`
- Match PyTorch/NumPy APIs when possible

## Key Files & Directories

- `tinygrad/tensor.py` - Tensor class, main user API
- `tinygrad/uop/ops.py` - UOp definition, the core IR
- `tinygrad/schedule/rangeify.py` - Convert tensor graph to kernels
- `tinygrad/engine/schedule.py` - Execution scheduling
- `tinygrad/codegen/opt/` - Kernel optimization passes
- `tinygrad/renderer/` - Code generation for each backend
- `tinygrad/runtime/ops_*.py` - Backend implementations
- `test/` - Test suite (avoid `extra/` - not well tested)
- `examples/` - Reference implementations (LLaMA, Stable Diffusion, etc.)

## Environment Variables Reference

Key variables for development:
- `DEBUG=[1-7]` - Debug output level
- `BEAM=#` - Kernel beam search width
- `JIT=[0-2]` - JIT compilation mode
- `IMAGE=[1-2]` - Enable 2D image optimizations
- Backend selection: `CUDA=1`, `METAL=1`, `AMD=1`, `CPU=1`, etc.

See `docs/env_vars.md` for complete list.

## Testing on Different Backends

```bash
# CPU with LLVM
CPU=1 CPU_LLVM=1 python3 test/test_ops.py

# CUDA
CUDA=1 python3 test/test_ops.py

# With beam search optimization
BEAM=2 python3 test/test_ops.py
```
