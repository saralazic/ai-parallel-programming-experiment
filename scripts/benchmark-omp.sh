#!/usr/bin/env bash
#
# benchmark-omp.sh — Run all OpenMP + sequential benchmarks with 10 repetitions
#                    per thread count, compute mean & stddev, generate Markdown report.
#
# Usage:
#   ./scripts/benchmark-omp.sh
#
# Output:
#   results/benchmark-omp-report.md
#
# Parameters used:
#   - Hotspot:    1024x1024 grid, 10000 iterations
#   - Moldyn:     20 particles
#   - Mandelbrot: default (500x500, 2000 max iter)
#   - Feynman-Kac: default (n=10000)
#   - SGEMM:     medium dataset (1024x1056 matrices)
#
# Thread counts tested: 1, 2, 4, 8, 10
# Repetitions per configuration: 10
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
REPORT="$RESULTS_DIR/benchmark-omp-report.md"
THREAD_COUNTS=(1 2 4 8 10)
REPS=10
MODELS=(codex opus gemini)
SGEMM_MODELS=(codex opus gemini reference)
HOTSPOT_MODELS=(codex opus gemini reference)
MANDELBROT_MODELS=(codex opus gemini reference)
MOLDYN_MODELS=(codex opus gemini reference)

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " OpenMP + Sequential Benchmark Suite"
echo " Repetitions: $REPS"
echo " Thread counts: ${THREAD_COUNTS[*]}"
echo " Models: ${MODELS[*]}"
echo "============================================"
echo ""

# --- Helper: compute mean and stddev from a file of numbers ---
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

# --- Detect environment info (runs inside Docker) ---
detect_environment() {
    local env_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
echo \"OS: \$(cat /etc/os-release | grep PRETTY_NAME | cut -d'\"' -f2)\"
echo \"GCC: \$(gcc --version | head -1)\"
echo \"CPU: \$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name: *//' || cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2 | xargs)\"
echo \"CORES: \$(nproc)\"
"
    docker run --rm ubuntu:22.04 bash -c "$env_script" 2>/dev/null
}

# --- Directory for validation results ---
VALIDATION_DIR=$(mktemp -d)

# ============================================================
# ACCURACY VALIDATION
# ============================================================

# Compare two files numerically with relative tolerance
# Usage: compare_numeric <file1> <file2> <tolerance>
# Returns 0 (pass) or 1 (fail)
compare_numeric() {
    local file1="$1" file2="$2" tol="$3"
    awk -v tol="$tol" '
    BEGIN { pass=1; lines=0 }
    NR==FNR { a[NR]=$0; n=NR; next }
    {
        lines++
        split(a[lines], ref_fields)
        split($0, test_fields)
        for (i=1; i<=length(ref_fields); i++) {
            r = ref_fields[i]+0
            t = test_fields[i]+0
            if (r == r && t == t) {
                if (r != 0) { if (((t-r)/r)^2 > tol^2) pass=0 }
                else if (t > tol) pass=0
            }
        }
    }
    END { print pass }
    ' "$file1" "$file2"
}

# Validate Mandelbrot: compare PPM output (exact pixel match)
validate_mandelbrot() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/mandelbrot/results/codex/code/openmp/mandelbrot-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/mandelbrot/results/opus/code/openmp/mandelbrot-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/mandelbrot/results/gemini/code/openmp/mandelbrot-gemini.c" ;;
        reference) src_file="$REPO_ROOT/problems/mandelbrot/reference/omp/mandelbrot.c" ;;
    esac
    local seq_src="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/mandelbrot_par.c"
    cp "$seq_src" "$work_dir/mandelbrot_seq.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
gcc -O2 -o mandelbrot_seq mandelbrot_seq.c -lm
gcc -O2 -fopenmp -o mandelbrot_par mandelbrot_par.c -lm
./mandelbrot_seq > /dev/null 2>&1
mv mandelbrot_openmp.ppm seq_output.ppm 2>/dev/null || mv mandelbrot.ppm seq_output.ppm 2>/dev/null || true
export OMP_NUM_THREADS=4
./mandelbrot_par > /dev/null 2>&1
mv mandelbrot_openmp.ppm par_output.ppm 2>/dev/null || mv mandelbrot.ppm par_output.ppm 2>/dev/null || true
if [ -f seq_output.ppm ] && [ -f par_output.ppm ]; then
    if diff -q seq_output.ppm par_output.ppm > /dev/null 2>&1; then
        echo 'VALIDATE result=Pass'
    else
        echo 'VALIDATE result=Fail'
    fi
else
    echo 'VALIDATE result=N/A'
fi
"
    local result
    result=$(docker run --rm -v "$work_dir:/work" -w /work ubuntu:22.04 bash -c "$docker_script" 2>/dev/null | grep "^VALIDATE")
    echo "$result" | sed 's/.*result=//'
    rm -rf "$work_dir"
}

# Validate Moldyn: compare stdout energy values with tolerance
validate_moldyn() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/moldyn/results/codex/code/openmp/moldyn-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/moldyn/results/opus/code/openmp/moldyn-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/moldyn/results/gemini/code/openmp/moldyn-gemini.c" ;;
        reference) src_file="$REPO_ROOT/problems/moldyn/reference/omp/moldyn.c" ;;
    esac
    local seq_src="$REPO_ROOT/problems/moldyn/original/moldyn-seq.c"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/moldyn_par.c"
    cp "$seq_src" "$work_dir/moldyn_seq.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
gcc -O2 -o moldyn_seq moldyn_seq.c -lm 2>/dev/null
gcc -O2 -fopenmp -o moldyn_par moldyn_par.c -lm 2>/dev/null
if [ ! -f moldyn_seq ] || [ ! -f moldyn_par ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi
./moldyn_seq 20 > seq_out.txt 2>/dev/null
export OMP_NUM_THREADS=4
./moldyn_par 20 > par_out.txt 2>/dev/null
# Extract energy values (columns 2,3,4 of output lines starting with numbers)
grep '^ *[0-9]' seq_out.txt | awk '{print \$2, \$3, \$4}' > seq_vals.txt
grep '^ *[0-9]' par_out.txt | awk '{print \$2, \$3, \$4}' > par_vals.txt
if [ ! -s seq_vals.txt ] || [ ! -s par_vals.txt ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi
# Compare with 1% relative tolerance
pass=1
paste seq_vals.txt par_vals.txt | while IFS= read -r line; do
    set -- \$line
    for i in 1 2 3; do
        ref=\$(echo \$line | awk -v i=\$i '{print \$i}')
        test=\$(echo \$line | awk -v i=\$((i+3)) '{print \$i}')
        if [ \"\$ref\" != \"\" ] && [ \"\$test\" != \"\" ]; then
            awk -v r=\"\$ref\" -v t=\"\$test\" 'BEGIN { if(r+0!=0 && ((t-r)/r)^2 > 0.01^2) exit 1 }' || { pass=0; break 2; }
        fi
    done
done
if [ \$pass -eq 1 ]; then
    echo 'VALIDATE result=Pass'
else
    echo 'VALIDATE result=Fail'
fi
"
    local result
    result=$(docker run --rm -v "$work_dir:/work" -w /work ubuntu:22.04 bash -c "$docker_script" 2>/dev/null | grep "^VALIDATE")
    echo "$result" | sed 's/.*result=//'
    rm -rf "$work_dir"
}

# Validate Feynman-Kac: compare RMS error from stdout
validate_feynman_kac() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/feynman-kac/results/codex/code/openmp/feynman-kac-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/feynman-kac/results/opus/code/openmp/feynman-kac-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/feynman-kac/results/gemini/code/openmp/feynman-kac-gemini.c" ;;
    esac
    local seq_src="$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/fk_par.c"
    cp "$seq_src" "$work_dir/fk_seq.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential > /dev/null 2>&1
gcc -O2 -o fk_seq fk_seq.c -lm 2>/dev/null
gcc -O2 -fopenmp -o fk_par fk_par.c -lm 2>/dev/null
if [ ! -f fk_seq ] || [ ! -f fk_par ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi
./fk_seq > seq_out.txt 2>/dev/null
export OMP_NUM_THREADS=4
./fk_par > par_out.txt 2>/dev/null
# Extract W Approx values from output
grep '^ *[0-9]' seq_out.txt | awk '{print \$4}' > seq_vals.txt
grep '^ *[0-9]' par_out.txt | awk '{print \$4}' > par_vals.txt
if [ ! -s seq_vals.txt ] || [ ! -s par_vals.txt ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi
# Compare with 5% relative tolerance (stochastic, parallel RNG may differ)
pass=\$(paste seq_vals.txt par_vals.txt | awk '
BEGIN { pass=1 }
{
    r=\$1+0; t=\$2+0;
    if (r!=0 && ((t-r)/r)^2 > 0.05^2) pass=0;
}
END { print pass }')
if [ \"\$pass\" = \"1\" ]; then
    echo 'VALIDATE result=Pass'
else
    echo 'VALIDATE result=Fail'
fi
"
    local result
    result=$(docker run --rm -v "$work_dir:/work" -w /work ubuntu:22.04 bash -c "$docker_script" 2>/dev/null | grep "^VALIDATE")
    echo "$result" | sed 's/.*result=//'
    rm -rf "$work_dir"
}

# Validate Hotspot: compare output temperature grid
# AI/seq write to output.txt when OUTPUT=1; Rodinia reference writes to argv[7]
validate_hotspot() {
    local model="$1"
    local src_file
    case "$model" in
        codex) src_file="$REPO_ROOT/problems/hotspot/results/codex/code/openmp/hotspot-codex.cpp" ;;
        opus)  src_file="$REPO_ROOT/problems/hotspot/results/opus/code/openmp/hotspot-opus.cpp" ;;
        gemini) src_file="$REPO_ROOT/problems/hotspot/results/gemini/code/openmp/hotspot-gemini.cpp" ;;
        reference) src_file="$REPO_ROOT/problems/hotspot/reference/omp/hotspot.cpp" ;;
    esac

    local data_dir="$REPO_ROOT/problems/hotspot/data"
    local seq_src="$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp"

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/hotspot_par.cpp"
    cp "$seq_src" "$work_dir/hotspot_seq.cpp"
    cp "$data_dir/temp_1024" "$work_dir/"
    cp "$data_dir/power_1024" "$work_dir/"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq g++ > /dev/null 2>&1
g++ -O2 -o hotspot_seq hotspot_seq.cpp 2>/dev/null
g++ -O2 -fopenmp -o hotspot_par hotspot_par.cpp 2>/dev/null
if [ ! -f hotspot_seq ] || [ ! -f hotspot_par ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi

# Sequential: OUTPUT=1 -> output.txt
export OUTPUT=1
./hotspot_seq 1024 1024 10000 1 temp_1024 power_1024 /dev/null > /dev/null 2>&1
mv output.txt seq_output.txt 2>/dev/null || true

export OMP_NUM_THREADS=4
if [ \"$model\" = \"reference\" ]; then
    # Rodinia reference always writes to argv[7]
    ./hotspot_par 1024 1024 10000 4 temp_1024 power_1024 par_output.txt > /dev/null 2>&1
else
    # AI models write to output.txt when OUTPUT=1
    ./hotspot_par 1024 1024 10000 4 temp_1024 power_1024 /dev/null > /dev/null 2>&1
    mv output.txt par_output.txt 2>/dev/null || true
fi

if [ ! -f seq_output.txt ] || [ ! -f par_output.txt ]; then
    echo 'VALIDATE result=N/A'
    exit 0
fi
# Compare temperature values with 1e-4 relative tolerance
pass=\$(paste seq_output.txt par_output.txt | awk '
BEGIN { pass=1 }
{
    r=\$2+0; t=\$4+0;
    if (r!=0 && ((t-r)/r)^2 > 0.0001^2) pass=0;
}
END { print pass }')
if [ \"\$pass\" = \"1\" ]; then
    echo 'VALIDATE result=Pass'
else
    echo 'VALIDATE result=Fail'
fi
"
    local result
    result=$(docker run --rm -v "$work_dir:/work" -w /work ubuntu:22.04 bash -c "$docker_script" 2>/dev/null | grep "^VALIDATE")
    echo "$result" | sed 's/.*result=//'
    rm -rf "$work_dir"
}

# Validate SGEMM: run through Parboil; built-in compare-output checks vs reference
# Tolerance (from tools/compare-output): abs(diff) < 0.01 OR < 1% of reference
validate_sgemm() {
    local model="$1"
    local src_dir
    if [[ "$model" == "reference" ]]; then
        src_dir="$REPO_ROOT/problems/sgemm/reference/omp"
    else
        src_dir="$REPO_ROOT/problems/sgemm/results/$model/code/openmp"
    fi
    local parboil_host="$HOME/Desktop/parboil"

    if [[ ! -d "$parboil_host" ]]; then
        echo "N/A"
        return
    fi

    local variant="omp_base"
    local dest_dir="$parboil_host/benchmarks/sgemm/src/$variant"
    mkdir -p "$dest_dir"
    cp "$src_dir"/*.cc "$dest_dir/" 2>/dev/null || true
    cp "$src_dir"/Makefile "$dest_dir/" 2>/dev/null || true

    # Ensure compare-output is executable
    chmod +x "$parboil_host/benchmarks/sgemm/tools/compare-output" 2>/dev/null || true

    local docker_script="
set +e
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential wget libreadline-dev libncurses5-dev \
    libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev libffi-dev libomp-dev \
    > /dev/null 2>&1

if [[ ! -f /usr/local/python2/bin/python2 ]]; then
    cd /tmp
    wget -q https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
    tar xf Python-2.7.18.tgz
    cd Python-2.7.18
    ./configure --prefix=/usr/local/python2 --enable-shared > /dev/null 2>&1
    make -j\$(nproc) > /dev/null 2>&1
    make install > /dev/null 2>&1
fi

# compare-output shebang uses 'python' — must be Python 2
ln -sf /usr/local/python2/bin/python2 /usr/local/bin/python
ln -sf /usr/local/python2/bin/python2 /usr/local/bin/python2
export PATH=/usr/local/python2/bin:/usr/local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/python2/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=/parboil/common/python:\${PYTHONPATH:-}
export PARBOIL_ROOT=/parboil
cd \$PARBOIL_ROOT

python2 ./parboil clean sgemm $variant > /dev/null 2>&1 || true
if ! python2 ./parboil compile sgemm $variant > /tmp/sgemm_compile.log 2>&1; then
    echo 'VALIDATE result=N/A'
    exit 0
fi

export OMP_NUM_THREADS=4
# parboil run checks output against datasets/sgemm/medium/output/matrix3.txt
run_out=\$(python2 ./parboil run sgemm $variant medium 2>&1)
run_rc=\$?

if echo \"\$run_out\" | grep -qi 'mismatch'; then
    echo 'VALIDATE result=Fail'
elif [ \$run_rc -eq 0 ]; then
    echo 'VALIDATE result=Pass'
else
    # Fallback: manual compare if run exited non-zero for other reasons
    ref=/parboil/datasets/sgemm/medium/output/matrix3.txt
    out=/parboil/benchmarks/sgemm/run/medium/matrix3.txt
    if [ -f \"\$ref\" ] && [ -f \"\$out\" ]; then
        if PYTHONPATH=/parboil/common/python python /parboil/benchmarks/sgemm/tools/compare-output \"\$ref\" \"\$out\" > /dev/null 2>&1; then
            echo 'VALIDATE result=Pass'
        else
            echo 'VALIDATE result=Fail'
        fi
    else
        echo 'VALIDATE result=N/A'
    fi
fi
"
    local result
    result=$(docker run --rm -v "$parboil_host:/parboil" ubuntu:22.04 bash -c "$docker_script" 2>/dev/null | grep "^VALIDATE" | tail -1)
    if [[ -n "$result" ]]; then
        echo "$result" | sed 's/.*result=//'
    else
        echo "N/A"
    fi
}

# ============================================================
# HOTSPOT
# ============================================================
benchmark_hotspot() {
    local model="$1"
    local src_file

    case "$model" in
        codex) src_file="$REPO_ROOT/problems/hotspot/results/codex/code/openmp/hotspot-codex.cpp" ;;
        opus)  src_file="$REPO_ROOT/problems/hotspot/results/opus/code/openmp/hotspot-opus.cpp" ;;
        gemini) src_file="$REPO_ROOT/problems/hotspot/results/gemini/code/openmp/hotspot-gemini.cpp" ;;
        reference) src_file="$REPO_ROOT/problems/hotspot/reference/omp/hotspot.cpp" ;;
    esac

    local data_dir="$REPO_ROOT/problems/hotspot/data"
    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/hotspot.cpp"
    cp "$data_dir/temp_1024" "$work_dir/"
    cp "$data_dir/power_1024" "$work_dir/"

    local thread_list="${THREAD_COUNTS[*]}"
    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq g++ bc > /dev/null 2>&1
g++ -O2 -fopenmp -o hotspot hotspot.cpp

for threads in $thread_list; do
    export OMP_NUM_THREADS=\$threads
    for i in \$(seq 1 $REPS); do
        result=\$( { time ./hotspot 1024 1024 10000 \$threads temp_1024 power_1024 /dev/null > /dev/null 2>&1 ; } 2>&1 | grep ^real | awk '{print \$2}' )
        mins=\$(echo \$result | sed 's/m.*//')
        secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
        total=\$(echo \"\$mins * 60 + \$secs\" | bc)
        echo \"RESULT threads=\$threads run=\$i time=\$total\"
    done
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

benchmark_hotspot_seq() {
    local src_file="$REPO_ROOT/problems/hotspot/original/hotspot-seq.cpp"
    local data_dir="$REPO_ROOT/problems/hotspot/data"
    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/hotspot.cpp"
    cp "$data_dir/temp_1024" "$work_dir/"
    cp "$data_dir/power_1024" "$work_dir/"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq g++ bc > /dev/null 2>&1
g++ -O2 -o hotspot hotspot.cpp

for i in \$(seq 1 $REPS); do
    result=\$( { time ./hotspot 1024 1024 10000 1 temp_1024 power_1024 /dev/null > /dev/null 2>&1 ; } 2>&1 | grep ^real | awk '{print \$2}' )
    mins=\$(echo \$result | sed 's/m.*//')
    secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
    total=\$(echo \"\$mins * 60 + \$secs\" | bc)
    echo \"RESULT threads=seq run=\$i time=\$total\"
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

# ============================================================
# MANDELBROT
# ============================================================
benchmark_mandelbrot() {
    local model="$1"
    local src_file

    case "$model" in
        codex) src_file="$REPO_ROOT/problems/mandelbrot/results/codex/code/openmp/mandelbrot-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/mandelbrot/results/opus/code/openmp/mandelbrot-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/mandelbrot/results/gemini/code/openmp/mandelbrot-gemini.c" ;;
        reference) src_file="$REPO_ROOT/problems/mandelbrot/reference/omp/mandelbrot.c" ;;
    esac

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/mandelbrot.c"

    local thread_list="${THREAD_COUNTS[*]}"
    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -fopenmp mandelbrot.c -lm -o mandelbrot

for threads in $thread_list; do
    export OMP_NUM_THREADS=\$threads
    for i in \$(seq 1 $REPS); do
        result=\$( { time ./mandelbrot ; } 2>&1 | grep ^real | awk '{print \$2}' )
        mins=\$(echo \$result | sed 's/m.*//')
        secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
        total=\$(echo \"\$mins * 60 + \$secs\" | bc)
        echo \"RESULT threads=\$threads run=\$i time=\$total\"
    done
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

benchmark_mandelbrot_seq() {
    local src_file="$REPO_ROOT/problems/mandelbrot/original/mandelbrot_seq.c"
    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/mandelbrot.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -o mandelbrot mandelbrot.c -lm

for i in \$(seq 1 $REPS); do
    result=\$( { time ./mandelbrot ; } 2>&1 | grep ^real | awk '{print \$2}' )
    mins=\$(echo \$result | sed 's/m.*//')
    secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
    total=\$(echo \"\$mins * 60 + \$secs\" | bc)
    echo \"RESULT threads=seq run=\$i time=\$total\"
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

# ============================================================
# MOLDYN
# ============================================================
benchmark_moldyn() {
    local model="$1"
    local src_file

    case "$model" in
        codex) src_file="$REPO_ROOT/problems/moldyn/results/codex/code/openmp/moldyn-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/moldyn/results/opus/code/openmp/moldyn-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/moldyn/results/gemini/code/openmp/moldyn-gemini.c" ;;
        reference) src_file="$REPO_ROOT/problems/moldyn/reference/omp/moldyn.c" ;;
    esac

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/moldyn.c"

    local thread_list="${THREAD_COUNTS[*]}"
    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -fopenmp moldyn.c -lm -o moldyn

for threads in $thread_list; do
    export OMP_NUM_THREADS=\$threads
    for i in \$(seq 1 $REPS); do
        result=\$( { time ./moldyn 20 ; } 2>&1 | grep ^real | awk '{print \$2}' )
        mins=\$(echo \$result | sed 's/m.*//')
        secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
        total=\$(echo \"\$mins * 60 + \$secs\" | bc)
        echo \"RESULT threads=\$threads run=\$i time=\$total\"
    done
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

benchmark_moldyn_seq() {
    local src_file="$REPO_ROOT/problems/moldyn/original/moldyn-seq.c"
    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/moldyn.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -o moldyn moldyn.c -lm

for i in \$(seq 1 $REPS); do
    result=\$( { time ./moldyn 20 ; } 2>&1 | grep ^real | awk '{print \$2}' )
    mins=\$(echo \$result | sed 's/m.*//')
    secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
    total=\$(echo \"\$mins * 60 + \$secs\" | bc)
    echo \"RESULT threads=seq run=\$i time=\$total\"
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

# ============================================================
# FEYNMAN-KAC
# ============================================================
benchmark_feynman_kac() {
    local model="$1"
    local src_file

    case "$model" in
        codex) src_file="$REPO_ROOT/problems/feynman-kac/results/codex/code/openmp/feynman-kac-codex.c" ;;
        opus)  src_file="$REPO_ROOT/problems/feynman-kac/results/opus/code/openmp/feynman-kac-opus.c" ;;
        gemini) src_file="$REPO_ROOT/problems/feynman-kac/results/gemini/code/openmp/feynman-kac-gemini.c" ;;
    esac

    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/feynman-kac.c"

    local thread_list="${THREAD_COUNTS[*]}"
    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -fopenmp feynman-kac.c -lm -o feynman-kac

for threads in $thread_list; do
    export OMP_NUM_THREADS=\$threads
    for i in \$(seq 1 $REPS); do
        result=\$( { time ./feynman-kac ; } 2>&1 | grep ^real | awk '{print \$2}' )
        mins=\$(echo \$result | sed 's/m.*//')
        secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
        total=\$(echo \"\$mins * 60 + \$secs\" | bc)
        echo \"RESULT threads=\$threads run=\$i time=\$total\"
    done
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

benchmark_feynman_kac_seq() {
    local src_file="$REPO_ROOT/problems/feynman-kac/original/feynman-kac-seq.c"
    local work_dir=$(mktemp -d)
    cp "$src_file" "$work_dir/feynman-kac.c"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
gcc -O2 -o feynman-kac feynman-kac.c -lm

for i in \$(seq 1 $REPS); do
    result=\$( { time ./feynman-kac ; } 2>&1 | grep ^real | awk '{print \$2}' )
    mins=\$(echo \$result | sed 's/m.*//')
    secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
    total=\$(echo \"\$mins * 60 + \$secs\" | bc)
    echo \"RESULT threads=seq run=\$i time=\$total\"
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

# ============================================================
# SGEMM (uses Parboil — needs special handling)
# ============================================================
benchmark_sgemm() {
    local model="$1"
    local src_dir
    if [[ "$model" == "reference" ]]; then
        src_dir="$REPO_ROOT/problems/sgemm/reference/omp"
    else
        src_dir="$REPO_ROOT/problems/sgemm/results/$model/code/openmp"
    fi
    local parboil_host="$HOME/Desktop/parboil"
    local variant="omp_base"

    if [[ ! -d "$parboil_host" ]]; then
        echo "  [SKIP] Parboil not found at $parboil_host"
        return
    fi

    local dest_dir="$parboil_host/benchmarks/sgemm/src/$variant"
    mkdir -p "$dest_dir"
    cp "$src_dir"/*.cc "$dest_dir/" 2>/dev/null || true
    cp "$src_dir"/Makefile "$dest_dir/" 2>/dev/null || true

    local thread_list="${THREAD_COUNTS[*]}"
    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential wget libreadline-dev libncurses5-dev \
    libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev libffi-dev libomp-dev bc \
    > /dev/null 2>&1

if [[ ! -f /usr/local/python2/bin/python2 ]]; then
    cd /tmp
    wget -q https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz
    tar xf Python-2.7.18.tgz
    cd Python-2.7.18
    ./configure --prefix=/usr/local/python2 --enable-shared > /dev/null 2>&1
    make -j\$(nproc) > /dev/null 2>&1
    make install > /dev/null 2>&1
fi

export PATH=/usr/local/python2/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/python2/lib:\${LD_LIBRARY_PATH:-}
export PARBOIL_ROOT=/parboil
cd \$PARBOIL_ROOT

python2 ./parboil clean sgemm $variant > /dev/null 2>&1 || true
python2 ./parboil compile sgemm $variant > /dev/null 2>&1

for threads in $thread_list; do
    export OMP_NUM_THREADS=\$threads
    for i in \$(seq 1 $REPS); do
        result=\$( { time python2 ./parboil run sgemm $variant medium ; } 2>&1 | grep ^real | awk '{print \$2}' )
        mins=\$(echo \$result | sed 's/m.*//')
        secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
        total=\$(echo \"\$mins * 60 + \$secs\" | bc)
        echo \"RESULT threads=\$threads run=\$i time=\$total\"
    done
done
"
    docker run --rm \
        -v "$parboil_host:/parboil" \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
}

benchmark_sgemm_seq() {
    local src_dir="$REPO_ROOT/problems/sgemm/original/base-without-parboil"
    local work_dir=$(mktemp -d)
    cp "$src_dir"/*.cc "$work_dir/"
    cp -r "$src_dir"/medium "$work_dir/"

    local docker_script="
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq build-essential bc > /dev/null 2>&1
g++ -O2 -o sgemm main.cc io.cc -lm

for i in \$(seq 1 $REPS); do
    result=\$( { time ./sgemm medium/input/matrix1.txt medium/input/matrix2t.txt medium/input/matrix2t.txt /dev/null ; } 2>&1 | grep ^real | awk '{print \$2}' )
    mins=\$(echo \$result | sed 's/m.*//')
    secs=\$(echo \$result | sed 's/.*m//' | sed 's/s//')
    total=\$(echo \"\$mins * 60 + \$secs\" | bc)
    echo \"RESULT threads=seq run=\$i time=\$total\"
done
"
    docker run --rm \
        -v "$work_dir:/work" -w /work \
        ubuntu:22.04 bash -c "$docker_script" 2>/dev/null
    rm -rf "$work_dir"
}

# ============================================================
# MAIN: Run all benchmarks and generate report
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
    model_list=("${MODELS[@]}")
    if [[ "$problem" == "sgemm" ]]; then
        model_list=("${SGEMM_MODELS[@]}")
    elif [[ "$problem" == "hotspot" ]]; then
        model_list=("${HOTSPOT_MODELS[@]}")
    elif [[ "$problem" == "mandelbrot" ]]; then
        model_list=("${MANDELBROT_MODELS[@]}")
    elif [[ "$problem" == "moldyn" ]]; then
        model_list=("${MOLDYN_MODELS[@]}")
    fi

    for model in "${model_list[@]}"; do
        echo ">>> Validating: $problem / $model"
        result=""
        case "$problem" in
            sgemm)       result=$(validate_sgemm "$model") ;;
            hotspot)     result=$(validate_hotspot "$model") ;;
            mandelbrot)  result=$(validate_mandelbrot "$model") ;;
            moldyn)      result=$(validate_moldyn "$model") ;;
            feynman-kac) result=$(validate_feynman_kac "$model") ;;
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
    echo ""
    echo ">>> Benchmarking: $problem (sequential)"
    
    raw_file="$RAW_DIR/${problem}_seq.txt"
    case "$problem" in
        sgemm)       benchmark_sgemm_seq > "$raw_file" ;;
        hotspot)     benchmark_hotspot_seq > "$raw_file" ;;
        mandelbrot)  benchmark_mandelbrot_seq > "$raw_file" ;;
        moldyn)      benchmark_moldyn_seq > "$raw_file" ;;
        feynman-kac) benchmark_feynman_kac_seq > "$raw_file" ;;
    esac
    echo "    Done."

    model_list=("${MODELS[@]}")
    if [[ "$problem" == "sgemm" ]]; then
        model_list=("${SGEMM_MODELS[@]}")
    elif [[ "$problem" == "hotspot" ]]; then
        model_list=("${HOTSPOT_MODELS[@]}")
    elif [[ "$problem" == "mandelbrot" ]]; then
        model_list=("${MANDELBROT_MODELS[@]}")
    elif [[ "$problem" == "moldyn" ]]; then
        model_list=("${MOLDYN_MODELS[@]}")
    fi

    for model in "${model_list[@]}"; do
        echo ">>> Benchmarking: $problem / $model (OpenMP)"
        
        raw_file="$RAW_DIR/${problem}_${model}.txt"
        case "$problem" in
            sgemm)       benchmark_sgemm "$model" > "$raw_file" ;;
            hotspot)     benchmark_hotspot "$model" > "$raw_file" ;;
            mandelbrot)  benchmark_mandelbrot "$model" > "$raw_file" ;;
            moldyn)      benchmark_moldyn "$model" > "$raw_file" ;;
            feynman-kac) benchmark_feynman_kac "$model" > "$raw_file" ;;
        esac
        echo "    Done."
    done
done

# --- Phase 3: Detect environment ---
echo ""
echo ">>> Detecting environment..."
ENV_INFO=$(detect_environment)
ENV_OS=$(echo "$ENV_INFO" | grep "^OS:" | sed 's/^OS: //')
ENV_GCC=$(echo "$ENV_INFO" | grep "^GCC:" | sed 's/^GCC: //')
# Host CPU (Docker lscpu often empty/generic on Apple Silicon)
if [[ "$(uname -s)" == "Darwin" ]]; then
    ENV_CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple M4")
    [[ -z "$ENV_CPU" ]] && ENV_CPU="Apple M4"
else
    ENV_CPU=$(echo "$ENV_INFO" | grep "^CPU:" | sed 's/^CPU: //')
fi
ENV_CORES=$(echo "$ENV_INFO" | grep "^CORES:" | sed 's/^CORES: //')

# ============================================================
# GENERATE MARKDOWN REPORT
# ============================================================

echo ""
echo ">>> Generating report: $REPORT"

cat > "$REPORT" << EOF
# OpenMP Benchmark Results

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| **Host OS** | macOS (Docker host) |
| **Container OS** | ${ENV_OS:-Ubuntu 22.04 LTS} |
| **CPU** | ${ENV_CPU:-N/A} |
| **CPU Cores** | ${ENV_CORES:-N/A} |
| **Compiler** | ${ENV_GCC:-gcc (Ubuntu)} |
| **Optimization flags** | \`-O2\` (all implementations) |
| **OpenMP flags** | \`-fopenmp\` |

## Methodology

- **Repetitions per configuration:** $REPS
- **Statistics:** arithmetic mean ± standard deviation (seconds)
- **Thread counts tested:** ${THREAD_COUNTS[*]}
- **Timing method:** bash \`time\` builtin (\`real\` wall-clock value)
- **Execution:** Each run in a fresh Docker ubuntu:22.04 container

## Problem Parameters

| Problem | Input / Size | Source |
|---------|-------------|--------|
| SGEMM | Medium dataset: 1024×1056 matrices | Parboil benchmark suite |
| Hotspot | 1024×1024 grid, 10000 iterations | Rodinia benchmark suite |
| Mandelbrot | 500×500 image, max 2000 iterations, region [-2.25,1.25]×[-1.75,1.75] | https://people.math.sc.edu/Burkardt/f77_src/mandelbrot_openmp/mandelbrot_openmp.html |
| Moldyn | mm=20 → npart=32000, 20 timesteps | EPCC Training Examples |
| Feynman-Kac | n=10000 paths, 21 grid points, a=2.0 | https://people.math.sc.edu/burkardt/c_src/feynman_kac_1d/feynman_kac_1d.html |

## Accuracy Verification

Parallel output compared against sequential baseline output.

| Problem | Tolerance | Comparison method |
|---------|-----------|-------------------|
| SGEMM | abs ≤ 0.01 or 1% relative | Parboil \`tools/compare-output\` vs reference matrix3.txt |
| Hotspot | 1e-4 relative | Temperature grid numerical comparison |
| Mandelbrot | exact | PPM image byte-for-byte comparison |
| Moldyn | 1% relative | Energy values (KE, PE, total) per timestep |
| Feynman-Kac | 5% relative | Approximate solution values (stochastic) |

EOF

for problem in "${PROBLEMS[@]}"; do
    echo "" >> "$REPORT"
    echo "---" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "## $problem" >> "$REPORT"
    echo "" >> "$REPORT"

    # Sequential result
    raw_file="$RAW_DIR/${problem}_seq.txt"
    if [[ -f "$raw_file" ]]; then
        times_file=$(mktemp)
        grep "RESULT threads=seq" "$raw_file" | sed 's/.*time=//' > "$times_file"
        stats=$(calc_stats "$times_file")
        seq_mean=$(echo "$stats" | awk '{print $1}')
        seq_std=$(echo "$stats" | awk '{print $2}')
        rm -f "$times_file"
        echo "**Sequential baseline:** ${seq_mean}s ± ${seq_std}s" >> "$REPORT"
        echo "" >> "$REPORT"
    fi

    # Per-model tables with accuracy
    report_models=("${MODELS[@]}")
    if [[ "$problem" == "sgemm" ]]; then
        report_models=("${SGEMM_MODELS[@]}")
    elif [[ "$problem" == "hotspot" ]]; then
        report_models=("${HOTSPOT_MODELS[@]}")
    elif [[ "$problem" == "mandelbrot" ]]; then
        report_models=("${MANDELBROT_MODELS[@]}")
    elif [[ "$problem" == "moldyn" ]]; then
        report_models=("${MOLDYN_MODELS[@]}")
    fi

    for model in "${report_models[@]}"; do
        raw_file="$RAW_DIR/${problem}_${model}.txt"
        if [[ ! -f "$raw_file" ]]; then
            continue
        fi

        accuracy=$(grep "^${problem}_${model} " "$ACCURACY_FILE" 2>/dev/null | awk '{print $2}')
        accuracy="${accuracy:-N/A}"
        echo "### $model (Accuracy: $accuracy)" >> "$REPORT"
        echo "" >> "$REPORT"

        # Per-model notes (intervention / compile issues)
        case "${problem}_${model}" in
            mandelbrot_opus)
                cat >> "$REPORT" <<'NOTE'
**Note:** Solution did not compile on the first attempt. An extra prompt was needed that only pasted the compiler error:

```
mandelbrot.c: In function 'main':
mandelbrot.c:55:5: error: not enough perfectly nested loops before 'y'
   55 |     y = ( ( double ) (     i - 1 ) * y_max
      |     ^
```

After that prompt, compilation succeeded.

NOTE
                ;;
            mandelbrot_codex)
                cat >> "$REPORT" <<'NOTE'
**Note:** Did not replace `clock()` with `omp_get_wtime()` on its own; only after an extra prompt: *"This result gives slower results with more threads than with 1 thread. What could be the issue?"*

NOTE
                ;;
            moldyn_opus)
                cat >> "$REPORT" <<'NOTE'
**Note:** Did not voluntarily change the timing function to use `omp_get_wtime()`; that change was applied manually.

NOTE
                ;;
            moldyn_codex)
                cat >> "$REPORT" <<'NOTE'
**Note:** Initial compile failed with:

```
moldyn.c: In function 'forces':
moldyn.c:161:7: error: 'stderr' not specified in enclosing 'parallel'
  161 |       fprintf(stderr, "Unable to allocate thread-local force buffer\n");
      |       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
moldyn.c:153:11: note: enclosing 'parallel'
  153 | #pragma omp parallel default(none)
      |           ^~~
```

Pasting this error as an extra prompt fixed the issue and compilation then succeeded.

NOTE
                ;;
        esac

        echo "| Threads | Mean (s) | Std Dev (s) | Speedup vs seq |" >> "$REPORT"
        echo "|---------|----------|-------------|----------------|" >> "$REPORT"

        for threads in "${THREAD_COUNTS[@]}"; do
            times_file=$(mktemp)
            grep "RESULT threads=$threads " "$raw_file" | sed 's/.*time=//' > "$times_file"
            stats=$(calc_stats "$times_file")
            mean=$(echo "$stats" | awk '{print $1}')
            std=$(echo "$stats" | awk '{print $2}')
            
            if [[ "$seq_mean" != "N/A" && "$mean" != "N/A" ]]; then
                speedup=$(echo "$seq_mean $mean" | awk '{if($2>0) printf "%.2fx", $1/$2; else print "N/A"}')
            else
                speedup="N/A"
            fi

            echo "| $threads | $mean | $std | $speedup |" >> "$REPORT"
            rm -f "$times_file"
        done

        echo "" >> "$REPORT"
    done
done

# Cleanup
rm -rf "$RAW_DIR"
rm -rf "$VALIDATION_DIR"
rm -f "$ACCURACY_FILE"

echo ""
echo "============================================"
echo " Benchmark complete!"
echo " Report: $REPORT"
echo "============================================"
