###############################################################
# Project: GPU Saturation Scorer
#
# File Name: metrics.py
#
# Description:
# This file contains the list of metrics to monitor by default.
#
# Authors:
# Marcel Ferrari (CSCS)
#
###############################################################


gpu_activity_metrics        = ["gpu_utilization", "sm_active", "sm_occupancy", "dram_active"]
flop_activity_metrics       = ["tensor_active", "fp64_active", "fp32_active", "fp16_active"]
memory_activity_metrics     = ["pcie_tx_bytes", "pcie_rx_bytes", "nvlink_tx_bytes", "nvlink_rx_bytes"]
all_metrics                 = gpu_activity_metrics + flop_activity_metrics + memory_activity_metrics

# List of field IDs to monitor
metric_ids = [
    203,    # DCGM_FI_DEV_GPU_UTIL
    250,    #Total framebuffer memory in MB
    251,    #Total framebuffer used in MB
    252,    #Total framebuffer free in MB
    253,    #Total framebuffer reserved in MB
    1002,   # DCGM_FI_PROF_SM_ACTIVE
    1003,   # DCGM_FI_PROF_SM_OCCUPANCY,
    1004,   # DCGM_FI_PROF_PIPE_TENSOR_CORE_ACTIVE
    1006,   # DCGM_FI_PROF_PIPE_FP64_ACTIVE
    1007,   # DCGM_FI_PROF_PIPE_FP32_ACTIVE
    1008,   # DCGM_FI_PROF_PIPE_FP16_ACTIVE
    1005,   # DCGM_FI_PROF_DRAM_ACTIVE
    1009,   # DCGM_FI_PROF_PCIE_TX_BYTES
    1010,   # DCGM_FI_PROF_PCIE_RX_BYTES
    1011,   # DCGM_FI_PROF_NVLINK_RX_BYTES
    1012    # DCGM_FI_PROF_NVLINK_TX_BYTES
]
