#!/bin/bash

export SLURM_PARTITION=debug
export SLURM_PARTITION=normal
export SLURM_RESERVATION=uss140-shs131-nv590-staging
export SBATCH_RESERVATION=$SLURM_RESERVATION

DCGMPROFTESTER=/usr/bin/dcgmproftester12
DCGMPROFTESTER=/usr/bin/true


make clean
make
make install-uv

mkdir -p testing-tmp
cd testing-tmp

GA=$(realpath ../gssr-analyze)
GR=$(realpath ../gssr-record)

set -x
#set -e

make -f ../Makefile clean-tests

function test_gr_basic()
{
    $GR -h
    $GR --help
    $GR --version
    $GR --help -o test-report-00
}

function test_ga_basic()
{
    $GA DOES_NOT_EXIST
    $GA test-report-01
    stat report.pdf

    mkdir -p test-report-01/nocluster_0/step_0
    $GA test-report-01

    touch test-report-01/nocluster_0/step_0/proc_0.meta.txt
    $GA test-report-01
}

function test_00_sleep()
{
    $GR -o -- sleep
    $GR -o test-report-00 sleep 5 && cat test-report-00/*/step_0/proc_0.{csv,meta.txt}
    $GR -o test-report-00 -- sleep 5 && cat test-report-00/*/step_0/proc_0.{csv,meta.txt}
}

function test_00_sleep_ga()
{
    $GR -o test-report-00 -- sleep 5
    $GA test-report-00 -o test-report-00.pdf
}

function test_00_dir_permission()
{
    $GR -o $SCRATCH/gssr-test -- sleep 1
    ls -ld $SCRATCH/gssr-test
}

function test_00_dir_name()
{
    $GR -o $SCRATCH/gssr-test -- sleep 1
    ls -ld $SCRATCH/gssr-test
}

function test_01_dcgmproftester()
{
    rm -rf test-report-01
    srun -N2 -n6 $GR -o test-report-01 $DCGMPROFTESTER -t 1006 -d 20 
    ls -ltr
    $GA test-report-01 -o test-report-01.pdf
}

function test_02_dcgmproftester()
{
    rm -rf test-report-02a
    rm -rf test-report-02b
    rm -rf test-report-02c

    srun -N1 -n1 -t 00:01:00 $GR -o test-report-02a $DCGMPROFTESTER -t 1006 -d 240 
    $GA test-report-02a -o test-report-02a.pdf

    srun -N1 -n4 --gpus-per-task=1 -t 00:01:00 $GR -o test-report-02b $DCGMPROFTESTER -t 1006 -d 240 
    $GA test-report-02b -o test-report-02b.pdf

    srun -N32 --ntasks-per-node=4 --gpus-per-task=1 -t 00:10:00 $GR -o test-report-02c $DCGMPROFTESTER -t 1006 -d 600  --max-processes 1
    $GA test-report-02c -o test-report-02c.pdf
}

function test_021_dcgmproftester()
{
    rm -rf test-report-021

    srun -N1 -n1 -t 00:05:00 $GR -o test-report-021 $DCGMPROFTESTER -t 1006,1007,1008,1013,1014,1015,1016 -d 20 
    $GA test-report-021 -o test-report-021.pdf
}

function test_01_long_args()
{
    rm -rf test-report-01
    srun -N2 -n6 $GR -o test-report-01 bash -lc '
        $DCGMPROFTESTER 
        -t "1006"
        -d '\''20'\''
        '

    ls -ltr
    $GA test-report-01 -o test-report-01.pdf
}

function test_020_dcgmproftester_128n()
{
    rm -rf test-report-020-large

    srun -N128 --ntasks-per-node=4 -t 00:010:00 $GR -o test-report-020-large $DCGMPROFTESTER -t 1006 -d 240 
    $GA test-report-020-large -o test-report-020.pdf
}

function test_01_multireport()
{
    rm -rf test-report-01
    srun -N2 -n6 $GR -o test-report-01 $DCGMPROFTESTER -t 1006 -d 20 
    srun -N2 -n6 $GR -o test-report-01 $DCGMPROFTESTER -t 1006 -d 20 
    srun -N2 -n6 $GR -o test-report-01 $DCGMPROFTESTER -t 1006 -d 20 
    ls -ltr test-report-01
    $GA test-report-01 -o test-report-01-multi.pdf
    for d in $(ls test-report-01); do
        $GA test-report-01/$d -o test-report-01.pdf
    done
}

function test_03_signal()
{
    rm -rf test-report-03
    srun -N1 -n1 --signal=HUP@30 -t 00:01:00 $GR -o test-report-03 $DCGMPROFTESTER -t 1006 -d 240
    $GA test-report-03 -o test-report-03.pdf
}

function test_04_long_running()
{
    rm -rf test-report-04
    srun -N3 -n3 -t 00:30:00 $GR -o test-report-04 $DCGMPROFTESTER -t 1006 -d 3600 
    $GA test-report-04 -o test-report-04.pdf
}

function test_05_sphexa()
{
    rm -rf dump_evrard.h5 dump_sedov.h5 test-report-05
    OMP_NUM_THREADS=12 time srun -N3 -n3 -c12 -t 00:03:00 --gpus-per-task=1 $GR -o test-report-05 sphexa/build/main/src/sphexa/sphexa-cuda --init evrard --glass 50c.h5 -n 200 -s 2000000 -w 2000000
    $GA test-report-05 -o test-report-05.pdf
}

function test_06_mps_wrapper()
{
    rm -rf test-report-06
    srun -N1 -n32 -t 00:05:00 ./mps-wrapper.sh $GR -o test-report-06 $DCGMPROFTESTER -t 1006 -d 60 
    $GA test-report-06 -o test-report-06.pdf
}

function test_07_multi_mps_wrapper()
{
    cat > test-report-07.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-07
#SBATCH -t 05:00
#SBATCH -N1
#SBATCH -n32
#SBATCH -A csstaff
#SBATCH -o test-report-07.slurm.out

srun ./mps-wrapper.sh $GR -o test-report-07 $DCGMPROFTESTER -t 1006 -d 60 
srun ./mps-wrapper.sh $GR -o test-report-07 $DCGMPROFTESTER -t 1007 -d 120 
EOF

    rm -rf test-report-07
    sbatch -W test-report-07.sh
    $GA test-report-07 -o test-report-07.pdf
}

function test_08_concurrent_srun()
{
    cat > test-report-08.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-08
#SBATCH -t 05:00
#SBATCH -N1
# BATCH -n4
#SBATCH -A csstaff
# BATCH --gpus-per-node=4
#SBATCH -o test-report-08.slurm.out
#SBATCH --exclusive --mem=450G


srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 $DCGMPROFTESTER --max-processes 1 -t 1006 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 $DCGMPROFTESTER --max-processes 1 -t 1007 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 $DCGMPROFTESTER --max-processes 1 -t 1008 -d 20 &
srun -N1 --ntasks-per-node=1 --exclusive --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-08 $DCGMPROFTESTER --max-processes 1 -t 1005 -d 20 &

wait
EOF

    sbatch -W test-report-08.sh
    $GA test-report-08 -o test-report-08.pdf
}

function test_09_overlapping_srun()
{
    cat > test-report-09.sh <<EOF
#!/bin/bash
#SBATCH -J test-report-09
#SBATCH -t 05:00
#SBATCH -N1
# BATCH -n4
#SBATCH -A csstaff
#SBATCH --gpus-per-node=1
#SBATCH -o test-report-09.slurm.out
#SBATCH --exclusive --mem=450G


srun --overlap -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-09 $DCGMPROFTESTER --max-processes 1 -t 1006 -d 20 &
srun --overlap -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=5 --mem=50G  $GR -o test-report-09 $DCGMPROFTESTER --max-processes 1 -t 1007 -d 20 &

wait
EOF

    sbatch -W test-report-09.sh
    $GA test-report-09 -o test-report-09.pdf
}

function test_10_container()
{
    cat > test-ubuntu.toml <<EOF
    image = "library/ubuntu:24.04"
    mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
    workdir = "${SCRATCH}"
EOF
    srun --environment=./test-ubuntu.toml $(realpath $GR) --help

    cat > test-ubuntu.toml <<EOF
    image = "library/ubuntu:24.04"
    mounts = ["${SCRATCH}:${SCRATCH}", "${HOME}:${HOME}"]
    workdir = "${SCRATCH}"

    [annotations]
    com.hooks.dcgm.enabled = "true"
EOF
    srun --environment=./test-ubuntu.toml $(realpath $GR) --help

}

function test_11_short_output()
{
    mkdir -p short-output/step_0
    python3 ../fake-csv.py -n 1 -g 1 -o short-output/step_0/proc_0.csv
    $GA short-output -o test-report-11a.pdf
    rm -rf short-output/step_0

    mkdir -p short-output/step_0
    python3 ../fake-csv.py -n 0 -g 1 -o short-output/step_0/proc_0.csv
    $GA short-output -o test-report-11b.pdf
    rm -rf short-output/step_0

    mkdir -p short-output/step_0
    python3 ../fake-csv.py -n 10 -g 0 -o short-output/step_0/proc_0.csv
    $GA short-output -o test-report-11c.pdf
    rm -rf short-output/step_0

    mkdir -p short-output/step_0
    python3 ../fake-csv.py -n 100 -g 1 -o short-output/step_0/proc_0.csv
    $GA short-output -o test-report-11d.pdf
    rm -rf short-output/step_0
}

function test_12_fake_output()
{
    mkdir -p fake-output/step_0
    python3 ../fake-csv.py -n 1000 -g 4 -o fake-output/step_0/proc_0.csv
    $GA fake-output -o test-report-12a.pdf
#   rm -rf fake-output/step_0

#   mkdir -p fake-output/step_0
#   python3 ../fake-csv.py -n 10000 -g 4 -o fake-output/step_0/proc_0.csv
#   $GA fake-output -o test-report-12b.pdf
#   rm -rf fake-output/step_0
}

function test_13_newlines_in_meta()
{
    mkdir -p test-report-13/nocluster_0/step_0

    cat > test-report-13/nocluster_0/step_0/proc_0.meta.txt <<-EOF
{
    "gssr-record-version": "2.0",
    "date": "2026-03-27T12:38:47+0100",
    "cluster": "clariden",
    "jobid": "1735344",
    "jobname": "test-grpo",
    "nnodes": "1",
    "ntasks": "1",
    "ngpus": "4",
    "step_nnodes": "1",
    "step_ntasks": "1",
    "executable": "bash",
    "arguments": "'-lc' '
       set -euo pipefail
       pip install -U datasets pillow peft bitsandbytes
       echo "[test] Starting inference with vLLM TP=\$TP_SIZE ..."
       CUDA_VISIBLE_DEVICES=0,1,2,3 python3 vlm_ground/testgrpo.py
     ' "
}
EOF
    $GA test-report-13
}

function test_13b_correct_arguments()
{
    mkdir -p test-report-13b/nocluster_0/step_0

    cat > test-report-13b/nocluster_0/step_0/proc_0.meta.txt <<-EOF
{
    "gssr-record-version": "2.0",
    "date": "2026-03-27T12:38:47+0100",
    "cluster": "clariden",
    "jobid": "1735344",
    "jobname": "test-grpo",
    "nnodes": "1",
    "ntasks": "1",
    "ngpus": "4",
    "step_nnodes": "1",
    "step_ntasks": "1",
    "executable": "bash",
    "arguments": [
        "-lc",
        "set -euo pipefail\\n pip install -U datasets pillow peft bitsandbytes\\n echo \\"[test] Starting inference with vLLM TP=\$TP_SIZE ...\\" CUDA_VISIBLE_DEVICES=0,1,2,3 python3 vlm_ground/testgrpo.py"
        ]
}
EOF
    $GA test-report-13b
}

function test_13c_correct_arguments()
{
    mkdir -p test-report-13c/nocluster_0/step_0

    cat > test-report-13c/nocluster_0/step_0/proc_0.meta.txt <<-"EOF"
{
    "gssr-record-version": "2.0",
    "date": "2026-04-02T18:44:51+0200",
    "cluster": "test-cluster",
    "jobid": "4567",
    "jobname": "test",
    "nnodes": "4",
    "ntasks": "128",
    "ngpus": "4",
    "step_nnodes": "2",
    "step_ntasks": "64",
    "executable": "escape_test",
    "arguments": [
      "a\nb",
      "a\rb",
      "a\tb",
      "\"double quotes\"",
      "'single quotes'",
      "\\\"\\\""
    ]
}
EOF
    $GA test-report-13c
}

function test_14_50util_check()
{
    mkdir -p fake-output/step_0
    python3 ../fake-csv-50util-check.py -n 10 -g 4 -o fake-output/step_0/proc_0.csv
    $GA fake-output -o test-report-14.pdf
    #rm -rf fake-output/step_0
}

#test_gr_basic
#test_ga_basic
#test_00_sleep
#test_00_dir_permission
#test_01_dcgmproftester
#test_01_multireport
#test_01_long_args
#test_02_dcgmproftester
#test_021_dcgmproftester
##test_020_dcgmproftester_128n
#test_03_signal
##test_04_long_running
##test_05_sphexa
##test_06_mps_wrapper
##test_07_multi_mps_wrapper
test_08_concurrent_srun
test_09_overlapping_srun
test_10_container
test_00_sleep_ga
test_11_short_output
test_12_fake_output
test_13_newlines_in_meta
test_13b_correct_arguments
test_13c_correct_arguments
test_01_long_args
test_14_50util_check
