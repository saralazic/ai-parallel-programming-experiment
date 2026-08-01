#!/usr/bin/env bash
#
# benchmark-cuda.sh — Run all CUDA + sequential benchmarks with 10 repetitions,
#                     compute mean & stddev, generate Markdown report.
#
# Usage (on the remote CUDA machine, from repo root):
#   ./scripts/benchmark-cuda.sh
#
# Output:
#   results/benchmark-cuda-report.md
#
# Parameters:
#   - Hotspot:    1024x1024 grid, 10000 iterations
#   - Moldyn:     20 particles
#   - Mandelbrot: 500x500 image, 2000 max iterations
#   - Feynman-Kac: n=10000 paths, 21 grid points
#   - SGEMM:     medium dataset (1024x1056 matrices)
#
# Repetitions per configuration: 10
#
# Timing methodology:
#   We use wall-clock time (bash `time` builtin, `real` value) for the entire
#   program execution. For CUDA programs this INCLUDES:
#     - Device memory allocation (cudaMalloc)
#     - Host-to-device memory transfers (cudaMemcpy H2D)
#     - Kernel execution
#     - Device-to-host memory transfers (cudaMemcpy D2H)
#     - Device memory deallocation (cudaFree)
#   This gives a fair end-to-end comparison with the sequential baseline.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
REPORT="$RESULTS_DIR/benchmark-cuda-report.md"
REPS=10
MODELS=(codex opus gemini)

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " CUDA + Sequential Benchmark Suite"
echo " Repetitions: $REPS"
echo " Models: ${MODELS[*]}"
echo "============================================"
echo ""

# --- Check for nvcc ---
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Run this on a CUDA-capable machine."
    exit 1
fi
echo "NVCC: $(nvcc --version | grep release)"
echo ""

# --- Helper: compute mean and stddev ---
calc_stats() {
    awk '{sum+=$1; sumsq+=$1*$1; n++} END {
        if(n>0) {
            mean=sum/n;
            if(n>1) stddev=sqrt((sumsq - sum*sum/n)/(n-1));
            else stddev=0;
            printf "%.4f %.4f", mean, stddev
        } else {
            printf "N/A N/A"
        }
    }' "$1"
}

# --- Helper: run binary N times and emit RESULT lines ---
run_n_times() {
    local binary="$1"
    local args="$2"
    local label="$3"

    for i in $(seq 1 $REPS); do
        result=$( { time $binary $args > /dev/null 2>&1 ; } 2>&1 | grep ^real | awk '{print $2}' )
        mins=$(echo $result | sed 's/m.*//')
        secs=$(echo $result | sed 's/.*m//' | sed 's/s//')
        total=$(echo "$mins * 60 + $secs" | bc)
        echo "RESULT type=$label run=$i time=$total"
    done
}

# ============================================================
# ACCURACY VALIDATION (CUDA)
# ============================================================

# Validate Mandelbrot CUDA vs sequential PPM (P3 or P6).
# GPU FMA / FP differences can flip a few boundary pixels, so Pass if
# >= 99% of RGB components match exactly.
validate_mandelbrot_cuda() {
    local model="$1"
    local src_file
    case "$model" in
        codex|opus|gemini)
            src_file="$REPO_ROOT/problems/mandelbrot/results/$model/code/cuda/mandelbrot_cuda.cu"
            ;;
        reference)
            src_file="$REPO_ROOT/problems/mandelbrot/reference/cuda/mandelbrot.cu"
            ;;
    esac
    local seq_src="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"

    if [[ ! -f "$src_file" ]]; then
        echo "N/A"
        return
    fi

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/mandelbrot_cuda.cu"
    cp "$seq_src" "$work_dir/mandelbrot_seq.c"
    cd "$work_dir"

    gcc -O2 -o mandelbrot_seq mandelbrot_seq.c -lm 2>/dev/null
    # Disable FMA contraction so GPU matches CPU iteration more closely
    nvcc -O2 -arch=sm_86 --fmad=false -o mandelbrot_cuda mandelbrot_cuda.cu -lm 2>/dev/null
    if [[ ! -f mandelbrot_cuda ]]; then
        nvcc -O2 -arch=sm_86 -o mandelbrot_cuda mandelbrot_cuda.cu -lm 2>/dev/null
    fi

    if [[ ! -f mandelbrot_seq ]] || [[ ! -f mandelbrot_cuda ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    ./mandelbrot_seq > /dev/null 2>&1
    if [[ -f mandelbrot_openmp.ppm ]]; then
        mv mandelbrot_openmp.ppm seq_output.ppm
    elif [[ -f mandelbrot.ppm ]]; then
        mv mandelbrot.ppm seq_output.ppm
    else
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    rm -f mandelbrot_cuda.ppm mandelbrot_openmp.ppm mandelbrot.ppm
    ./mandelbrot_cuda > /dev/null 2>&1

    local cuda_ppm=""
    for f in mandelbrot_cuda.ppm mandelbrot_openmp.ppm mandelbrot.ppm; do
        [[ -f "$f" ]] && cuda_ppm="$f" && break
    done

    if [[ -z "$cuda_ppm" ]] || [[ ! -f seq_output.ppm ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    # Full compare in one Python process (handles P3/P6, reports Pass/Fail)
    local result
    result=$(python3 - seq_output.ppm "$cuda_ppm" <<'PY'
import sys

def read_ppm_rgb(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        def tokens():
            while True:
                line = f.readline()
                if not line:
                    return
                if line.startswith(b"#"):
                    continue
                for t in line.split():
                    yield t
        tok = tokens()
        w = int(next(tok)); h = int(next(tok)); _maxv = int(next(tok))
        n = w * h * 3
        if magic == b"P3":
            vals = []
            for t in tok:
                vals.append(int(t))
                if len(vals) >= n:
                    break
            return vals[:n]
        if magic == b"P6":
            raw = f.read(n)
            if len(raw) < n:
                return None
            return list(raw)
        return None

a = read_ppm_rgb(sys.argv[1])
b = read_ppm_rgb(sys.argv[2])
if not a or not b or len(a) != len(b) or len(a) == 0:
    print("N/A")
    sys.exit(0)
diff = sum(1 for x, y in zip(a, b) if x != y)
match_ratio = 1.0 - (diff / float(len(a)))
# 99% component match → Pass (boundary FP differences on GPU)
print("Pass" if match_ratio >= 0.99 else "Fail")
PY
)

    echo "${result:-N/A}"
    cd "$REPO_ROOT"; rm -rf "$work_dir"
}

# Validate Moldyn CUDA: compare energy values with tolerance
validate_moldyn_cuda() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/moldyn/results/codex/code/cuda/moldyn.cu" ;;
        opus)  src_file="$REPO_ROOT/problems/moldyn/results/opus/code/cuda/moldyn.cu" ;;
        gemini) src_file="$REPO_ROOT/problems/moldyn/results/gemini/code/cuda/moldyn.cu" ;;
    esac
    local seq_src="$REPO_ROOT/problems/moldyn/original/moldyn-seq.c"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/moldyn.cu"
    cp "$seq_src" "$work_dir/moldyn_seq.c"
    cd "$work_dir"

    gcc -O2 -o moldyn_seq moldyn_seq.c -lm 2>/dev/null
    nvcc -O2 -arch=sm_86 -o moldyn_cuda moldyn.cu -lm 2>/dev/null

    if [[ ! -f moldyn_seq ]] || [[ ! -f moldyn_cuda ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    ./moldyn_seq 20 > seq_out.txt 2>/dev/null
    ./moldyn_cuda 20 > cuda_out.txt 2>/dev/null

    grep '^ *[0-9]' seq_out.txt | awk '{print $2, $3, $4}' > seq_vals.txt
    grep '^ *[0-9]' cuda_out.txt | awk '{print $2, $3, $4}' > cuda_vals.txt

    if [[ ! -s seq_vals.txt ]] || [[ ! -s cuda_vals.txt ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    local pass
    pass=$(paste seq_vals.txt cuda_vals.txt | awk '
    BEGIN { pass=1 }
    {
        for (i=1; i<=3; i++) {
            r=$(i)+0; t=$(i+3)+0;
            if (r!=0 && ((t-r)/r)^2 > 0.01^2) pass=0;
        }
    }
    END { print pass }')

    if [[ "$pass" == "1" ]]; then echo "Pass"; else echo "Fail"; fi
    cd "$REPO_ROOT"; rm -rf "$work_dir"
}

# Validate Hotspot CUDA: compare output temperature values
#
# AI models → vs educational sequential (same single-step solver + same step size).
#
# Rodinia CUDA reference → N/A (not Pass/Fail). Upstream Rodinia CUDA uses
#   step = PRECISION/max_slope
# while Rodinia OpenMP / educational sequential use
#   step = PRECISION/max_slope/1000.0
# So CUDA↔OMP (and CUDA↔seq) grids diverge by design; Rodinia has no golden
# checker for Hotspot. Timing still uses the stock Rodinia CUDA binary.
validate_hotspot_cuda() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/hotspot/results/codex/code/cuda/hotspot-cuda.cu" ;;
        opus)  src_file="$REPO_ROOT/problems/hotspot/results/opus/code/cuda/hotspot-cuda.cu" ;;
        gemini) src_file="$REPO_ROOT/problems/hotspot/results/gemini/code/cuda/hotspot-cuda.cu" ;;
        reference)
            # Known Rodinia timestep mismatch vs OMP/seq — skip numerical check
            echo "N/A"
            return
            ;;
    esac
    local seq_src="$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp"
    local data_dir="$REPO_ROOT/problems/hotspot/data"

    if [[ ! -f "$src_file" ]]; then
        echo "N/A"
        return
    fi

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/hotspot_cuda.cu"
    cp "$seq_src" "$work_dir/hotspot_seq.cpp"
    cp "$data_dir/temp_1024" "$work_dir/"
    cp "$data_dir/power_1024" "$work_dir/"
    cd "$work_dir"

    g++ -O2 -o hotspot_seq hotspot_seq.cpp 2>/dev/null
    nvcc -O2 -arch=sm_86 -o hotspot_cuda hotspot_cuda.cu 2>/dev/null

    if [[ ! -f hotspot_seq ]] || [[ ! -f hotspot_cuda ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    export OUTPUT=1
    ./hotspot_seq 1024 1024 10000 1 temp_1024 power_1024 /dev/null > /dev/null 2>&1
    mv output.txt baseline_output.txt 2>/dev/null || true

    ./hotspot_cuda 1024 1024 10000 1 temp_1024 power_1024 /dev/null > /dev/null 2>&1
    mv output.txt cuda_output.txt 2>/dev/null || true

    if [[ ! -f baseline_output.txt ]] || [[ ! -f cuda_output.txt ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    # abs ≤ 0.1°C or 1% relative
    local pass
    pass=$(paste baseline_output.txt cuda_output.txt | awk '
    function abs(x) { return x < 0 ? -x : x }
    BEGIN { pass=1; n=0 }
    {
        n++
        r=$2+0; t=$4+0
        d=abs(r-t)
        if (!(d < 0.1 || d < 0.01 * abs(r))) pass=0
    }
    END { if (n==0) print 0; else print pass }')

    if [[ "$pass" == "1" ]]; then echo "Pass"; else echo "Fail"; fi
    cd "$REPO_ROOT"; rm -rf "$work_dir"
}

# Validate Feynman-Kac CUDA
validate_feynman_kac_cuda() {
    local model="$1"
    local src_file extra_libs=""
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/feynman-kac/results/codex/code/cuda/feynman-kac-gpu.cu" ;;
        opus)  src_file="$REPO_ROOT/problems/feynman-kac/results/opus/code/cuda/feynman-kac-cuda.cu"; extra_libs="-lcurand" ;;
        gemini) src_file="$REPO_ROOT/problems/feynman-kac/results/gemini/code/cuda/feynman-kac-cuda.cu"; extra_libs="-lcurand" ;;
    esac
    local seq_src="$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/fk_cuda.cu"
    cp "$seq_src" "$work_dir/fk_seq.c"
    cd "$work_dir"

    gcc -O2 -o fk_seq fk_seq.c -lm 2>/dev/null
    nvcc -O2 -arch=sm_86 -o fk_cuda fk_cuda.cu -lm $extra_libs 2>/dev/null

    if [[ ! -f fk_seq ]] || [[ ! -f fk_cuda ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    ./fk_seq > seq_out.txt 2>/dev/null
    ./fk_cuda > cuda_out.txt 2>/dev/null

    grep '^ *[0-9]' seq_out.txt | awk '{print $4}' > seq_vals.txt
    grep '^ *[0-9]' cuda_out.txt | awk '{print $4}' > cuda_vals.txt

    if [[ ! -s seq_vals.txt ]] || [[ ! -s cuda_vals.txt ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    local pass
    pass=$(paste seq_vals.txt cuda_vals.txt | awk '
    BEGIN { pass=1 }
    {
        r=$1+0; t=$2+0;
        if (r!=0 && ((t-r)/r)^2 > 0.05^2) pass=0;
    }
    END { print pass }')

    if [[ "$pass" == "1" ]]; then echo "Pass"; else echo "Fail"; fi
    cd "$REPO_ROOT"; rm -rf "$work_dir"
}

# Validate SGEMM CUDA: compare output matrices
# Output format: "rows cols val1 val2 ...\n" (space-separated floats)
# Tolerance: abs(diff) < 0.01 OR relative < 1% (Parboil compare-output)
validate_sgemm_cuda() {
    local model="$1"
    local src_dir="$REPO_ROOT/problems/sgemm/original/base-without-parboil"
    local cuda_dir

    case "$model" in
        codex|opus|gemini)
            cuda_dir="$REPO_ROOT/problems/sgemm/results/$model/code/cuda"
            ;;
        reference_base)
            cuda_dir="$REPO_ROOT/problems/sgemm/reference/cuda"
            ;;
        reference_optimized)
            cuda_dir="$REPO_ROOT/problems/sgemm/reference/cuda_optimized"
            ;;
    esac

    if [[ ! -d "$cuda_dir" ]]; then
        echo "N/A"
        return
    fi

    local work_dir=$(mktemp -d)
    mkdir -p "$work_dir/cuda_src"
    # Sequential source + data
    cp "$src_dir"/*.cc "$work_dir/" 2>/dev/null || true
    cp -r "$src_dir"/medium "$work_dir/"
    # CUDA sources only (skip nested medium/ dirs in model folders)
    find "$cuda_dir" -maxdepth 1 -type f \( -name '*.cc' -o -name '*.cu' -o -name '*.h' -o -name 'Makefile' \) \
        -exec cp {} "$work_dir/cuda_src/" \;
    cp -r "$work_dir/medium" "$work_dir/cuda_src/"
    cd "$work_dir"

    g++ -O2 -o sgemm_seq main.cc io.cc -lm 2>/dev/null
    if [[ ! -f sgemm_seq ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi
    ./sgemm_seq medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt seq_output.txt > /dev/null 2>&1

    cd cuda_src
    local binary=""

    # Reference standalone builds: compile all sources explicitly (most reliable)
    if [[ "$model" == "reference_base" || "$model" == "reference_optimized" ]]; then
        nvcc -O2 -arch=sm_86 -o sgemm main.cu io.cc sgemm_kernel.cu -lm >/dev/null 2>&1
        [[ -f ./sgemm ]] && binary="sgemm"
    fi

    if [[ -z "$binary" ]] || [[ ! -f "./$binary" ]]; then
        if grep -q "^cuda:" Makefile 2>/dev/null; then
            make cuda NVCCFLAGS="-O2 -arch=sm_86" >/dev/null 2>&1
            binary=$(grep "^CUDA_TARGET" Makefile | awk -F= '{print $2}' | tr -d ' ')
            [[ -z "$binary" ]] && binary="sgemm_cuda"
        else
            make NVCCFLAGS="-O2 -arch=sm_86" >/dev/null 2>&1
            binary=$(grep "^TARGET" Makefile | head -1 | awk -F= '{print $2}' | tr -d ' ')
            [[ -z "$binary" ]] && binary="sgemm"
        fi
    fi
    if [[ ! -f "./$binary" ]]; then
        binary=$(ls sgemm sgemm_cuda 2>/dev/null | head -1)
    fi
    # Last-resort nvcc
    if [[ -z "$binary" ]] || [[ ! -f "./$binary" ]]; then
        nvcc -O2 -arch=sm_86 -o sgemm main.cu io.cc sgemm_kernel.cu -lm >/dev/null 2>&1 \
            || nvcc -O2 -arch=sm_86 -o sgemm *.cu *.cc -lm >/dev/null 2>&1
        [[ -f ./sgemm ]] && binary="sgemm"
    fi

    if [[ -z "$binary" ]] || [[ ! -f "./$binary" ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    ./$binary medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt cuda_output.txt > /dev/null 2>&1
    if [[ ! -f cuda_output.txt ]] || [[ ! -f ../seq_output.txt ]]; then
        echo "N/A"
        cd "$REPO_ROOT"; rm -rf "$work_dir"
        return
    fi

    # Compare all floats (same length, Parboil-like tolerance)
    local pass
    pass=$(awk '
    function abs(x) { return x < 0 ? -x : x }
    BEGIN { pass=1 }
    FNR==NR { for (i=1; i<=NF; i++) ref[++nref]=$i+0; next }
    {
        ncmp=0
        for (i=1; i<=NF; i++) cmp[++ncmp]=$i+0
    }
    END {
        if (nref != ncmp || nref == 0) { print 0; exit }
        for (i=1; i<=nref; i++) {
            r=ref[i]; c=cmp[i]; d=abs(r-c)
            if (!(d < 0.01 || d < 0.01 * abs(r))) pass=0
        }
        print pass
    }' ../seq_output.txt cuda_output.txt)

    if [[ "$pass" == "1" ]]; then echo "Pass"; else echo "Fail"; fi
    cd "$REPO_ROOT"; rm -rf "$work_dir"
}

# ============================================================
# SGEMM
# ============================================================
benchmark_sgemm() {
    echo ">>> SGEMM sequential..."
    local src_dir="$REPO_ROOT/problems/sgemm/original/base-without-parboil"
    local work_dir=$(mktemp -d)
    cp "$src_dir"/*.cc "$work_dir/"
    cp -r "$src_dir"/medium "$work_dir/"
    cd "$work_dir"
    g++ -O2 -o sgemm main.cc io.cc -lm
    run_n_times "./sgemm" "medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt /dev/null" "seq"
    cd "$REPO_ROOT"
    rm -rf "$work_dir"

    for model in "${MODELS[@]}"; do
        echo ">>> SGEMM CUDA $model..."
        local cuda_dir="$REPO_ROOT/problems/sgemm/results/$model/code/cuda"
        if [[ ! -d "$cuda_dir" ]]; then
            echo "  [SKIP] Not found: $cuda_dir"
            continue
        fi
        local work_dir=$(mktemp -d)
        find "$cuda_dir" -maxdepth 1 -type f \( -name '*.cc' -o -name '*.cu' -o -name '*.h' -o -name 'Makefile' \) \
            -exec cp {} "$work_dir/" \;
        cp -r "$src_dir"/medium "$work_dir/" 2>/dev/null || true
        cd "$work_dir"
        local compile_err binary

        if grep -q "^cuda:" Makefile 2>/dev/null; then
            compile_err=$(make cuda NVCCFLAGS="-O2 -arch=sm_86" 2>&1)
            binary=$(grep "^CUDA_TARGET" Makefile | awk -F= '{print $2}' | tr -d ' ')
            [[ -z "$binary" ]] && binary="sgemm_cuda"
        else
            compile_err=$(make NVCCFLAGS="-O2 -arch=sm_86" 2>&1)
            binary=$(grep "^TARGET" Makefile | head -1 | awk -F= '{print $2}' | tr -d ' ')
            [[ -z "$binary" ]] && binary="sgemm"
        fi

        if [[ -f "./$binary" ]]; then
            run_n_times "./$binary" "medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt /dev/null" "cuda_$model"
        else
            echo "ERROR model=$model stage=compile msg=$(echo "$compile_err" | tail -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done

    # Reference CUDA implementations (base and optimized)
    for ref_variant in base optimized; do
        echo ">>> SGEMM CUDA reference_$ref_variant..."
        local ref_dir="$REPO_ROOT/problems/sgemm/reference/cuda"
        [[ "$ref_variant" == "optimized" ]] && ref_dir="$REPO_ROOT/problems/sgemm/reference/cuda_optimized"

        if [[ ! -d "$ref_dir" ]]; then
            echo "  [SKIP] Not found: $ref_dir"
            continue
        fi
        local work_dir=$(mktemp -d)
        find "$ref_dir" -maxdepth 1 -type f \( -name '*.cc' -o -name '*.cu' -o -name '*.h' -o -name 'Makefile' \) \
            -exec cp {} "$work_dir/" \;
        cp -r "$src_dir"/medium "$work_dir/" 2>/dev/null || true
        cd "$work_dir"
        local compile_err

        compile_err=$(make NVCCFLAGS="-O2 -arch=sm_86" 2>&1)

        if [[ -f "./sgemm" ]]; then
            run_n_times "./sgemm" "medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt /dev/null" "cuda_reference_$ref_variant"
        else
            echo "ERROR model=reference_$ref_variant stage=compile msg=$(echo "$compile_err" | tail -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done
}

# ============================================================
# HOTSPOT
# ============================================================
benchmark_hotspot() {
    local data_dir="$REPO_ROOT/problems/hotspot/data"

    if [[ ! -f "$data_dir/temp_1024" ]]; then
        echo "  [SKIP] Hotspot data files not found in $data_dir"
        return
    fi

    echo ">>> Hotspot sequential..."
    local work_dir=$(mktemp -d)
    cp "$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp" "$work_dir/hotspot.cpp"
    cp "$data_dir/temp_1024" "$work_dir/"
    cp "$data_dir/power_1024" "$work_dir/"
    cd "$work_dir"
    g++ -O2 -o hotspot hotspot.cpp
    run_n_times "./hotspot" "1024 1024 10000 1 temp_1024 power_1024 /dev/null" "seq"
    cd "$REPO_ROOT"
    rm -rf "$work_dir"

    for model in "${MODELS[@]}"; do
        echo ">>> Hotspot CUDA $model..."
        local src_file="$REPO_ROOT/problems/hotspot/results/$model/code/cuda/hotspot-cuda.cu"
        if [[ ! -f "$src_file" ]]; then
            echo "  [SKIP] Not found: $src_file"
            continue
        fi
        local work_dir=$(mktemp -d)
        cp "$src_file" "$work_dir/hotspot-cuda.cu"
        cp "$data_dir/temp_1024" "$work_dir/"
        cp "$data_dir/power_1024" "$work_dir/"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o hotspot hotspot-cuda.cu 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./hotspot" "1024 1024 10000 1 temp_1024 power_1024 /dev/null" "cuda_$model"
        else
            echo "ERROR model=$model stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done

    # Reference CUDA (Rodinia) — different interface: <grid> <pyramid_h> <iterations> <temp> <power>
    echo ">>> Hotspot CUDA reference..."
    local ref_file="$REPO_ROOT/problems/hotspot/reference/cuda/hotspot.cu"
    if [[ -f "$ref_file" ]]; then
        local work_dir=$(mktemp -d)
        cp "$ref_file" "$work_dir/hotspot.cu"
        cp "$data_dir/temp_1024" "$work_dir/"
        cp "$data_dir/power_1024" "$work_dir/"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o hotspot hotspot.cu 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./hotspot" "1024 1 10000 temp_1024 power_1024" "cuda_reference"
        else
            echo "ERROR model=reference stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    fi
}

# ============================================================
# MANDELBROT
# ============================================================
benchmark_mandelbrot() {
    echo ">>> Mandelbrot sequential..."
    local work_dir=$(mktemp -d)
    cp "$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c" "$work_dir/mandelbrot.c"
    cd "$work_dir"
    gcc -O2 -o mandelbrot mandelbrot.c -lm
    run_n_times "./mandelbrot" "" "seq"
    cd "$REPO_ROOT"
    rm -rf "$work_dir"

    for model in "${MODELS[@]}"; do
        echo ">>> Mandelbrot CUDA $model..."
        local src_file="$REPO_ROOT/problems/mandelbrot/results/$model/code/cuda/mandelbrot_cuda.cu"
        if [[ ! -f "$src_file" ]]; then
            echo "  [SKIP] Not found: $src_file"
            continue
        fi
        local work_dir=$(mktemp -d)
        cp "$src_file" "$work_dir/mandelbrot_cuda.cu"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o mandelbrot mandelbrot_cuda.cu 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./mandelbrot" "" "cuda_$model"
        else
            echo "ERROR model=$model stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done

    # Mandelbrot CUDA reference
    echo ">>> Mandelbrot CUDA reference..."
    local ref_src="$REPO_ROOT/problems/mandelbrot/reference/cuda/mandelbrot.cu"
    if [[ -f "$ref_src" ]]; then
        local work_dir=$(mktemp -d)
        cp "$ref_src" "$work_dir/mandelbrot.cu"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o mandelbrot mandelbrot.cu -lm 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./mandelbrot" "" "cuda_reference"
        else
            echo "ERROR model=reference stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    else
        echo "  [SKIP] Reference CUDA not found: $ref_src"
    fi
}

# ============================================================
# MOLDYN
# ============================================================
benchmark_moldyn() {
    echo ">>> Moldyn sequential..."
    local work_dir=$(mktemp -d)
    cp "$REPO_ROOT/problems/moldyn/original/moldyn-seq.c" "$work_dir/moldyn.c"
    cd "$work_dir"
    gcc -O2 -o moldyn moldyn.c -lm
    run_n_times "./moldyn" "20" "seq"
    cd "$REPO_ROOT"
    rm -rf "$work_dir"

    for model in "${MODELS[@]}"; do
        echo ">>> Moldyn CUDA $model..."
        local src_file
        case "$model" in
            codex) src_file="$REPO_ROOT/problems/moldyn/results/codex/code/cuda/moldyn.cu" ;;
            opus)  src_file="$REPO_ROOT/problems/moldyn/results/opus/code/cuda/moldyn.cu" ;;
            gemini) src_file="$REPO_ROOT/problems/moldyn/results/gemini/code/cuda/moldyn.cu" ;;
        esac
        if [[ ! -f "$src_file" ]]; then
            echo "  [SKIP] Not found: $src_file"
            continue
        fi
        local work_dir=$(mktemp -d)
        cp "$src_file" "$work_dir/moldyn.cu"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o moldyn moldyn.cu -lm 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./moldyn" "20" "cuda_$model"
        else
            echo "ERROR model=$model stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done
}

# ============================================================
# FEYNMAN-KAC
# ============================================================
benchmark_feynman_kac() {
    echo ">>> Feynman-Kac sequential..."
    local work_dir=$(mktemp -d)
    cp "$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c" "$work_dir/feynman-kac.c"
    cd "$work_dir"
    gcc -O2 -o feynman-kac feynman-kac.c -lm
    run_n_times "./feynman-kac" "" "seq"
    cd "$REPO_ROOT"
    rm -rf "$work_dir"

    for model in "${MODELS[@]}"; do
        echo ">>> Feynman-Kac CUDA $model..."
        local src_file extra_libs=""
        case "$model" in
            codex) src_file="$REPO_ROOT/problems/feynman-kac/results/codex/code/cuda/feynman-kac-gpu.cu" ;;
            opus)  src_file="$REPO_ROOT/problems/feynman-kac/results/opus/code/cuda/feynman-kac-cuda.cu"; extra_libs="-lcurand" ;;
            gemini) src_file="$REPO_ROOT/problems/feynman-kac/results/gemini/code/cuda/feynman-kac-cuda.cu"; extra_libs="-lcurand" ;;
        esac
        if [[ ! -f "$src_file" ]]; then
            echo "  [SKIP] Not found: $src_file"
            continue
        fi
        local work_dir=$(mktemp -d)
        cp "$src_file" "$work_dir/feynman-kac-cuda.cu"
        cd "$work_dir"
        local compile_err
        compile_err=$(nvcc -O2 -arch=sm_86 -o feynman-kac feynman-kac-cuda.cu -lm $extra_libs 2>&1)
        if [[ $? -eq 0 ]]; then
            run_n_times "./feynman-kac" "" "cuda_$model"
        else
            echo "ERROR model=$model stage=compile msg=$(echo "$compile_err" | head -5 | tr '\n' ' ')"
        fi
        cd "$REPO_ROOT"
        rm -rf "$work_dir"
    done

}

# ============================================================
# MAIN
# ============================================================

RAW_DIR=$(mktemp -d)
PROBLEMS=(sgemm hotspot mandelbrot moldyn feynman-kac)

# --- Phase 1: Accuracy Validation ---
echo ""
echo "============================================"
echo " Phase 1: Accuracy Validation"
echo "============================================"
echo ""

ACCURACY_FILE=$(mktemp)

for problem in "${PROBLEMS[@]}"; do
    all_models=("${MODELS[@]}")
    if [[ "$problem" == "sgemm" ]]; then
        all_models+=("reference_base" "reference_optimized")
    elif [[ "$problem" == "hotspot" ]]; then
        all_models+=("reference")
    elif [[ "$problem" == "mandelbrot" ]]; then
        all_models+=("reference")
    fi

    for model in "${all_models[@]}"; do
        echo ">>> Validating: $problem / $model (CUDA)"
        result=""
        case "$problem" in
            sgemm)       result=$(validate_sgemm_cuda "$model") ;;
            hotspot)     result=$(validate_hotspot_cuda "$model") ;;
            mandelbrot)  result=$(validate_mandelbrot_cuda "$model") ;;
            moldyn)      result=$(validate_moldyn_cuda "$model") ;;
            feynman-kac) result=$(validate_feynman_kac_cuda "$model") ;;
        esac
        result="${result:-N/A}"
        echo "${problem}_${model} ${result}" >> "$ACCURACY_FILE"
        echo "    -> ${result}"
    done
done

# --- Phase 2: Performance Benchmarks ---
echo ""
echo "============================================"
echo " Phase 2: Performance Benchmarks"
echo "============================================"
echo ""

for problem in "${PROBLEMS[@]}"; do
    raw_file="$RAW_DIR/${problem}.txt"
    case "$problem" in
        sgemm)       benchmark_sgemm > "$raw_file" ;;
        hotspot)     benchmark_hotspot > "$raw_file" ;;
        mandelbrot)  benchmark_mandelbrot > "$raw_file" ;;
        moldyn)      benchmark_moldyn > "$raw_file" ;;
        feynman-kac) benchmark_feynman_kac > "$raw_file" ;;
    esac
    echo "  -> $problem done."
done

# --- Phase 3: Detect environment ---
echo ""
echo ">>> Detecting environment..."
ENV_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
ENV_CUDA=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5 $6}' | tr -d ',' || echo "N/A")
ENV_GCC=$(gcc --version 2>/dev/null | head -1 || echo "N/A")
ENV_OS=$(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || uname -s)
ENV_CPU=$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name: *//' || echo "N/A")

# ============================================================
# GENERATE MARKDOWN REPORT
# ============================================================

echo ""
echo ">>> Generating report: $REPORT"

cat > "$REPORT" << EOF
# CUDA Benchmark Results

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| **OS** | ${ENV_OS} |
| **CPU** | ${ENV_CPU} |
| **GPU** | ${ENV_GPU} |
| **CUDA Toolkit** | ${ENV_CUDA} |
| **NVCC flags** | \`-O2 -arch=sm_86\` |
| **GCC (sequential)** | ${ENV_GCC} |
| **GCC flags** | \`-O2\` |

## Methodology

- **Repetitions per configuration:** $REPS
- **Statistics:** arithmetic mean ± standard deviation (seconds)
- **Timing:** wall-clock time (\`real\` from bash \`time\` builtin) for the entire program execution
- **GPU architecture target:** sm_86 (compute capability 8.6)

### What CUDA timing includes

The measured time is **end-to-end wall-clock** time, which includes:
- Device memory allocation (\`cudaMalloc\`)
- Host-to-device memory transfers (\`cudaMemcpy H→D\`)
- Kernel execution (GPU computation)
- Device-to-host memory transfers (\`cudaMemcpy D→H\`)
- Device memory deallocation (\`cudaFree\`)
- Any CPU-side setup within the program

This gives a fair total-cost comparison with the sequential CPU baseline.

## Problem Parameters

| Problem | Input / Size | Source |
|---------|-------------|--------|
| SGEMM | Medium dataset: 1024×1056 matrices | Parboil benchmark suite |
| Hotspot | 1024×1024 grid, 10000 iterations | Rodinia benchmark suite |
| Mandelbrot | 500×500 image, max 2000 iterations, region [-2.25,1.25]×[-1.75,1.75] | https://people.math.sc.edu/Burkardt/f77_src/mandelbrot_openmp/mandelbrot_openmp.html |
| Moldyn | mm=20 → npart=32000, 20 timesteps | EPCC Training Examples |
| Feynman-Kac | n=10000 paths, 21 grid points, a=2.0 | https://people.math.sc.edu/burkardt/c_src/feynman_kac_1d/feynman_kac_1d.html |

## Accuracy Verification

CUDA output compared against a CPU baseline (see notes below).

| Problem | Tolerance | Comparison method |
|---------|-----------|-------------------|
| SGEMM | abs ≤ 0.01 or 1% relative | Output matrix vs sequential (Parboil-like) |
| Hotspot (AI models) | abs ≤ 0.1°C or 1% relative | Temperature grid vs educational sequential |
| Hotspot (Rodinia reference) | N/A | Not numerically comparable (see note) |
| Mandelbrot | ≥99% RGB match | PPM pixels (P3/P6); allows GPU boundary FP diffs |
| Moldyn | 1% relative | Energy values (KE, PE, total) per timestep |
| Feynman-Kac | 5% relative | Approximate solution values (stochastic) |

**Note — Hotspot Rodinia CUDA reference:** Accuracy is reported as **N/A**, not
Fail. Upstream Rodinia CUDA computes the integration step as
`PRECISION/max_slope`, while Rodinia OpenMP and the educational sequential code
use `PRECISION/max_slope/1000.0`. That 1000× timestep difference makes CUDA↔OMP
(and CUDA↔seq) temperature grids diverge by design; Rodinia ships no golden-file
checker for Hotspot. Timing still uses the stock Rodinia CUDA binary. AI Hotspot
solutions keep the `/1000.0` formulation and are validated against sequential.

EOF

for problem in "${PROBLEMS[@]}"; do
    raw_file="$RAW_DIR/${problem}.txt"
    if [[ ! -f "$raw_file" ]]; then
        continue
    fi

    echo "" >> "$REPORT"
    echo "---" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "## $problem" >> "$REPORT"
    echo "" >> "$REPORT"

    # Sequential stats
    times_file=$(mktemp)
    grep "RESULT type=seq " "$raw_file" | sed 's/.*time=//' > "$times_file"
    stats=$(calc_stats "$times_file")
    seq_mean=$(echo "$stats" | awk '{print $1}')
    seq_std=$(echo "$stats" | awk '{print $2}')
    rm -f "$times_file"

    echo "**Sequential baseline (CPU):** ${seq_mean}s ± ${seq_std}s" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Model | Accuracy | Mean (s) | Std Dev (s) | Speedup vs seq |" >> "$REPORT"
    echo "|-------|----------|----------|-------------|----------------|" >> "$REPORT"

    # Determine which labels to report
    all_labels=()
    for model in "${MODELS[@]}"; do
        all_labels+=("cuda_$model")
    done
    if [[ "$problem" == "sgemm" ]]; then
        all_labels+=("cuda_reference_base" "cuda_reference_optimized")
    elif [[ "$problem" == "hotspot" ]]; then
        all_labels+=("cuda_reference")
    elif [[ "$problem" == "mandelbrot" ]]; then
        all_labels+=("cuda_reference")
    fi

    for label in "${all_labels[@]}"; do
        display_name="${label#cuda_}"
        accuracy=$(grep "^${problem}_${display_name} " "$ACCURACY_FILE" 2>/dev/null | awk '{print $2}')
        accuracy="${accuracy:-N/A}"
        times_file=$(mktemp)
        grep "RESULT type=$label " "$raw_file" | sed 's/.*time=//' > "$times_file"

        if [[ -s "$times_file" ]]; then
            stats=$(calc_stats "$times_file")
            mean=$(echo "$stats" | awk '{print $1}')
            std=$(echo "$stats" | awk '{print $2}')

            if [[ "$seq_mean" != "N/A" && "$mean" != "N/A" ]]; then
                speedup=$(echo "$seq_mean $mean" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')
            else
                speedup="N/A"
            fi
            echo "| $display_name | $accuracy | $mean | $std | $speedup |" >> "$REPORT"
        else
            echo "| $display_name | $accuracy | **ERROR** | — | — |" >> "$REPORT"
        fi
        rm -f "$times_file"
    done

    # Per-problem/model intervention notes
    case "$problem" in
        hotspot)
            cat >> "$REPORT" <<'NOTE'

**Note (codex):** COMPILATION FAILED — the same solution was inserted twice into the file. The duplicate half was deleted manually; after that, compilation succeeded.
NOTE
            ;;
    esac

    # Append any errors as notes
    errors=$(grep "^ERROR" "$raw_file" 2>/dev/null || true)
    if [[ -n "$errors" ]]; then
        echo "" >> "$REPORT"
        echo "**Runtime / compile errors:**" >> "$REPORT"
        while IFS= read -r line; do
            err_model=$(echo "$line" | sed 's/.*model=\([^ ]*\).*/\1/')
            err_stage=$(echo "$line" | sed 's/.*stage=\([^ ]*\).*/\1/')
            err_msg=$(echo "$line" | sed 's/.*msg=//')
            echo "- \`$err_model\`: $err_stage failed — \`$err_msg\`" >> "$REPORT"
        done <<< "$errors"
    fi

    echo "" >> "$REPORT"
done

# Cleanup
rm -rf "$RAW_DIR"
rm -f "$ACCURACY_FILE"

echo ""
echo "============================================"
echo " Benchmark complete!"
echo " Report: $REPORT"
echo "============================================"
