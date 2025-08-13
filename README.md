# GPU Saturation Scorer (gssr)
`gssr` is a utility meant to collect and analyze GPU performance metrics on the CSCS ALPS System. it is based on top of Nvidia's DCGM tool.

## Install
### From Pypi
```
pip install gssr
```

### From GitHub Source
```
pip install git+https://github.com/eth-cscs/GPU-saturation-scorer.git
```
To install from a specific branch, e.g. the development branch
```
pip install git+https://github.com/eth-cscs/GPU-saturation-scorer.git@dev
```
To install a specific release from a tag, e.g. gssr-v0.3
```
pip install git+https://github.com/eth-cscs/GPU-saturation-scorer.git@gssr-v0.3
```

## Profile
### Example
If you are submitting a batch job and the command you are executing is 
```
srun python test.py
```
The srun command should be modified as follows.:
```
srun gssr profile -wrap="python abc.py"
```
* The agi option to run is "profile".
* The "---wrap" flag will wrap the command you would like to run.
* The default output directory is "profile_out_{job_id}"
* You can also set a label to this output data if you prefer with the "-l" flag

## Analyze
### Metric Output
The profiled output can be analysed as follows.:
```
gssr analyze -i ./profile_out
```
### PDF File Output with Plots
```
gssr analyze -i ./profile_out --report
```
A/Multiple PDF report(s) will be generated containing all the generated plots.

### Exporting the Profiled Output as a SQLite3 file
```
gssr analyze -i ./profile_out --export data.sqlite3
```
## More Options
```
gssr --help
```

