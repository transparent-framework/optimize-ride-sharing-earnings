### Instructions to use this framework

#### Step 1: 
Update the directory structure under the `app` section of the YAML config file.

#### Step 2: 
Based upon your needs, please set the RL parameters under the `RL_parameters` section of the config file.

#### Step 3: 
Uncomment any single job / experiment that you wish to run in the `jobs` and `experiments` sections of the config file.

#### Step 4: 
Use `./sh/run_app.sh` to trigger a job.

### Note: 
Before running jobs or experiments, you will need to run all the initialization jobs included in the config file
in order to create all the relevant data structures required during the training of framework or experiments.

A more detailed README on instructions would be included after the reviewing process is finished, because it requires using
scripts and hosting supplemental datasets that may potentially break the anonymity required for a double-blind submission.

Cheers!
