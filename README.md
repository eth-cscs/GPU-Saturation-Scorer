# ALPS GPU Insight – AGI
AGI is a utility meant to collect and analyze GPU performance metrics on the CSCS ALPS System. it is based on top of Nvidia's DCGM tool.

## Install
```
pip install git+https://github.com/eth-cscs/MLp-system-performances-analysis-tool.git
```
To install from a specific branch, e.g. the development branch
```
pip install git+https://github.com/eth-cscs/MLp-system-performances-analysis-tool.git@dev
```
To install a specific release from a tag, e.g. v0.1.1
```
pip install git+https://github.com/eth-cscs/MLp-system-performances-analysis-tool.git@v0.1.1
```

## Profile
### Example
If you are submitting a batch job and the command you are executing is 
```
srun python test.py
```
The srun command should be modified as follows.:
```
srun agi profile -o ./profile_out -l my_job --wrap="python abc.py"
```
* The agi option to run is "profile".
* The "-o" flag represents the directory where you would like agi to output the profiled data to.
* The "-l" flag represents the label you would like to set for this output data.
* The "---wrap" flag will wrap the command you would like to run. 

## Analyze
### Metric Output
The profiled output can be analysed as follows.:
```
agi analyze -i ./profile_out
```
### PDF File Output with Plots
```
agi analyze -i ./profile_out --report
```
A/Multiple PDF report(s) will be generated containing all the generated plots.

### Exporting the Profiled Output as a SQLite3 file
```
agi analyze -i ./profile_out --export data.sqlite3
```


