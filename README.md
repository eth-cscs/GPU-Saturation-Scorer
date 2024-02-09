# ALPS GPU Insight – AGI
AGI is a utility meant to collect and analyze GPU performance metrics on the CSCS ALPS System.

### Requirements
AGI currently depends on the following python packages:
- numpy
- pandas
- scikit-learn
- filelock

You can install the required packages via `pip`:
```
pip install -r requirements.txt
```

Furthermore, AGI depends on the Nvidia DCGM utility to collect GPU metrics. By default, AGI will try to look into the default installation directory `/usr/local/dcgm` in order to load the required Python bindings. If DCGM is installed in a custom folder, set the environment variable `$DCGM_HOME` to tell AGI where to look.

### Profiling with AGI
Profiling a GPU workload with AGI is straightforward and is done by invoking the `profile` module:
```
srun python AGI.py profile -o myMetrics.sql "myCommand arg1 arg2"
```
This will profile your and output the collected metrics in an SQLite database.

The profiling module currently supports the following options:

```
--max-runtime max-runtime, -m max-runtime
                    Maximum runtime of the wrapped command in seconds

--sampling-time sampling-time, -s sampling-time
                    Sampling time of GPU metrics in milliseconds

--verbose, -v       Print verbose GPU metrics to stdout

--force-overwrite, -f
                    Force overwrite of output file

--output-file output-file, -o output-file
                    Output SQL file for collected GPU metrics
```

### Analyzing the output of AGI
There are two main ways to analyze the output of the AGI profile module: the first is to use the analysis module provided in AGI, which processes the data in order to output simple to intepret metrics. This can be done by invoking the module either directly on the cluster, or on your local machine:
```
python AGI.py analyze --detect-outliers leading --verbose --input-file="myMetrics.sql"
```
The analysis module currently supports the following options:

```
--input-file INPUT_FILE, -i INPUT_FILE
                        Input file for analysis

--verbose, -v         Print verbose GPU metrics to stdout

--detect-outliers, -d {leading,trailing,none,all}  
                        Heuristically detect outlier samples and discard them from the analysis
```

The second method is to access the raw data directly by reading the SQLite library using for example the `sqlite3` python module or the [PandaSQLite](https://github.com/MarcelFerrari/PandaSQLite) library.