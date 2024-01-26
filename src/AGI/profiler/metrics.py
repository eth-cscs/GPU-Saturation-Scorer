import dcgm_fields
import numpy as np

# List of field IDs to monitor and corresponding demangled names

metricIds = [dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
            dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE,
            dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES,
            dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES
            ]

demangledMetricNames = {
    dcgm_fields.DCGM_FI_DEV_GPU_UTIL: "DEV_GPU_UTIL",
    dcgm_fields.DCGM_FI_PROF_SM_ACTIVE: "SM_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY: "SM_OCCUPANCY",
    dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE: "PIPE_TENSOR_CORE_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_PIPE_FP64_ACTIVE: "PIPE_FP64_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_PIPE_FP32_ACTIVE: "PIPE_FP32_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_PIPE_FP16_ACTIVE: "PIPE_FP16_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_DRAM_ACTIVE: "DRAM_ACTIVE",
    dcgm_fields.DCGM_FI_PROF_PCIE_TX_BYTES: "PCIE_TX_BYTES",
    dcgm_fields.DCGM_FI_PROF_PCIE_RX_BYTES: "PCIE_RX_BYTES"
}

# Downsample a 1D array by averaging to n samples
def downsampleMetrics(data, n):
    # Special case: n <= 0 or len(data) = 0 -> return empty list
    if n <= 0 or len(data) == 0:
        return []
    # Special case: n = 1 or len(data) = 1 -> there is no downsampling to be done
    elif n == 1 or len(data) == n:
        return data

    # Split the array into n segments
    segments = np.array_split(np.array(data), n)

    # Compute the mean of each segment
    downsampled = [segment.mean() for segment in segments]

    return downsampled