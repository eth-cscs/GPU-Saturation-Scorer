import numpy as np

# List of metric names to monitor
gpu_activity_metrics = ["DEV_GPU_UTIL",
                        "SM_ACTIVE", "SM_OCCUPANCY", "DRAM_ACTIVE"]
flop_activity_metrics = ["PIPE_TENSOR_CORE_ACTIVE",
                         "PIPE_FP64_ACTIVE", "PIPE_FP32_ACTIVE", "PIPE_FP16_ACTIVE"]
memory_activity_metrics = ["PCIE_TX_BYTES", "PCIE_RX_BYTES"]
all_metrics = gpu_activity_metrics + \
    flop_activity_metrics + memory_activity_metrics

# List of field IDs to monitor
metric_ids = [
    203,    # DCGM_FI_DEV_GPU_UTIL
    1002,   # DCGM_FI_PROF_SM_ACTIVE
    1003,   # DCGM_FI_PROF_SM_OCCUPANCY
    1004,   # DCGM_FI_PROF_PIPE_TENSOR_CORE_ACTIVE
    1006,   # DCGM_FI_PROF_PIPE_FP64_ACTIVE
    1007,   # DCGM_FI_PROF_PIPE_FP32_ACTIVE
    1008,   # DCGM_FI_PROF_PIPE_FP16_ACTIVE
    1005,   # DCGM_FI_PROF_DRAM_ACTIVE
    1009,   # DCGM_FI_PROF_PCIE_TX_BYTES
    1010    # DCGM_FI_PROF_PCIE_RX_BYTES
]

# Map of field IDs to metric names
demangled_metric_names = {
    203: 'DEV_GPU_UTIL',
    1002: 'SM_ACTIVE',
    1003: 'SM_OCCUPANCY',
    1004: 'PIPE_TENSOR_CORE_ACTIVE',
    1006: 'PIPE_FP64_ACTIVE',
    1007: 'PIPE_FP32_ACTIVE',
    1008: 'PIPE_FP16_ACTIVE',
    1005: 'DRAM_ACTIVE',
    1009: 'PCIE_TX_BYTES',
    1010: 'PCIE_RX_BYTES'}
