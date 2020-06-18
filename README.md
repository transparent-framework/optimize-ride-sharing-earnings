### Visualization videos

#### Degree of coordination evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/coordination_probability.mp4?raw=true)

#### Coordinated wait action evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/wait_probability.mp4?raw=true)

#### Popular relocation zones during coordination: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/relocation_probability.mp4?raw=true)

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

#### Instructions to install the OpenAI reinforcement learning environment - nyc-yellow-taxi-v0 for testing deep learning based baselines

```
cd gym_nyc_yellow_taxi
pip install -e .
```
Once the new environment is installed, you may use it as follows:

```
import gym
import gym_nyc_yellow_taxi

env_id = "nyc-yellow-taxi-v0"
env = gym.make(env_id, config_=config)
```

where the `config` is loaded from `gym_nyc_yellow_taxi/config/gym_config.yaml`. The config file allows the user to change the number of drivers in the environment, and their initial distribution across the city.

A more detailed README on instructions would be included after the reviewing process is finished, because it requires using
scripts that may potentially break the anonymity required for a double-blind submission.

Cheers!
