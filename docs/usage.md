# Usage
This document details the usage and workflow for the GSSR - GPU saturation scorer profiling utility.

## Typical Workflow
The typical workflow to profile a workload consists of two main steps:

1. Collecting performance data from the GPUs for a given workload
2. Analysing and visualising the performance results

GSSR divides these two steps by defining two modules named `profile` and `analyze`.

### Profiling a workload
The first data collection step is done via the `profile module`. Currently GSSR only supports profiling workloads launched via the SLURM workload manager. This is necessary in order to obtain information regarding which resources must be monitored. GSSR should be called _after_ the `srun` command. Moreover, it is best practice to set the number of GPUs per task esplicitly by passing `--gpus-per-task=1`. It is important to notice that currently **GSSR only supports workloads that allocate 1 GPU per CPU (MPI) process**. In the future, support for different configurations will be added.

An profiling command for a workload named `my_workload` could look like this:

```bash
srun -N 4 --ntasks-per-node=4 --gpus-per-task=1 gssr profile -o my_workload.sql --wrap "./my_workload arg1 arg2 ... argN"
```

Notice the inclusion of `-o my_workload.sql`: GSSR stores the collected data in a SQLite dabase. The easiest way to access the raw metrics as well as the metadata is with the `sqlite3` python module combined with the `read_sql` method of the Pandas library.

The workload launch commad is wrapped in quotaation marks and passed to the `--wrap` argument, similarly to how the `sbatch --wrap` option works.

A more detailed explanation of the options provided by the `profile` module can be found in the corresponding markdown file.

### Analyzing performance results

Once the SQL database file containing the performance data is available, the `analyze` module can be used to quickly and conveniently generate more intuitive representations of the data. Specifically, there are three main aggregation strategies that can be exploited:

1. Aggregation over space: with this strategy, the time series data collected over all GPUs (space dimension) is aggregated to form an "average" time-series. This information is really useful in order to analyze the average usage over all GPUs as a function of time. This visualisation can provide insights on how the resources are used over the lifespan of a workload. 
2. Aggregation over time: this approach reduces the time-series data to a single point for each GPU by computing the time average of each metric over the monitored time interval. The result is a dataset that offers insights on the distribution of the resource usage over all GPUs. This is particularly interesting when analyzing a workload that is partitioned _unevenly_ and that can likely suffer from load balancing issues.
3. Aggregation over space and time: in this last case, both the space and time dimensions are reduced to a single pointwise value. This strategy is the most likely to suffer from information loss, however it is still a useful starting point in order to quickly evaluate the performance of a workload without generating additional visualisations.

All aggregation techniques are accessed via the `analyze` module by specifying different flags. By default, only aggregation over space and time is carried out.

An example analysis command that applies all three aggregation techniques is:

```
gssr analyze -i my_workload.sql --plot-time-series --plot-load-balancing
```
or alternatively:
```
gssr analyze -i my_workload.sql -pts -plb
```

These commands will automatically generate a set of graphical visualisations for the collected GPU metrics.

A more detailed explanation of the options provided by the `analyze` module can be found in the corresponding markdown file.

## Performance metrics

GSSR collects different profiling metrics that can help in evaluating both the efficiency as well as the inefficiencies of a workload. The metrics offered by GSSR are the following:
1. SM Activity
2. SM Occupancy
3. Tensor Activity
4. FP64/32/16 and FLOP Engine Activity
5. DRAM activity
6. PCIe Bandwidth
7. Workload efficiency score

The file `metrics.pdf`offers a more detailed overview and explanation of many of these metrics.

The `FLOP_ACTIVE` metric simply refers to the sum of the activities of each mathematical engine. I.e.: $$\text{FLOP ACTIVE} =  \text{FP64 ACTIVE} + \text{FP32 ACTIVE} + \text{FP16 ACTIVE} + \text{TENSOR ACTIVE}$$

The workload efficiency score is computed by an EOS (equation of state) that has been fitted on a set of synthetic data:
$$S = 4[\mathrm{sigmoid}(\alpha A)-0.5]\cdot[\mathrm{sigmoid}(\beta F + \gamma O \cdot e^{- \lambda F }) - 0.5]$$

Where:

* $S$ is the efficiency score
* $A$ is the SM Activity
* $F$ is the total FLOP activity
* $O$ is the occupancy
* $\alpha, \beta, \gamma, \lambda$ are learned parameters

The efficiency score is supposed to be an indicative estimate of "how efficiently a certain workload utilizes the available resources". It cannot be used to compare two different algorithms/workloads and should not be used as a replacement for time-to-solution.
