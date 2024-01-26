# ALPS GPU Insight – AGI
AGI is a utility meant to collect and analyze GPU performance metrics on the CSCS ALPS System.

### Requirements
AGI depends on the Nvidia DCGM utility to collect GPU metrics. By default, AGI will try to look into the default installation directory `/usr/local/dcgm` in order to load the required Python bindings. If DCGM is installed in a custom folder, set the variable `$DCGM_HOME` to tell AGI where to look.