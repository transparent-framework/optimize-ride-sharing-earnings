### Visualization videos

#### Degree of coordination evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/coordination_probability.mp4?raw=true)
In this framework, for lack of a better alternative, at the start of the day all the drivers are uniformly distributed across
the greater New York City. As a result, degree of coordination is high all across the city (except Manhattan which still has demand at midnight) facilitating the movement of drivers from all across the city to neighborhoods with the demand. As the day progresses, we observe that coordination is required in varying degrees in 3 major spots viz., downtown Manhattan and the two airports. We see a smooth increase/decrease of coordination over geographical region and time, thereby confirming that our framework is learning supply-demand characteristics as they evolve.

#### Coordinated wait action evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/wait_probability.mp4?raw=true)
Based on the need of coordination, in this video we show the probability that the recommended coordination action to the drivers is waiting in their current location. Interestingly, we find that probability of coordinated wait action is distributed
all across the greater NYC in the morning rush hours (movement towards Manhattan from outside boroughs). As the day progresses, the coordinated wait is limited to Manhattan and sometimes at the 2 airports. As the evening approaches, we see that the region for coordinated wait action gradually expands to cover neighborhoods outside downtown Manhattan.

#### Popular relocation zones during coordination: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/relocation_probability.mp4?raw=true)
Based on the need of coordination, in this video we attempt to show the popularity of relocation targets. As each relocation ride has a source and destination, it is difficult to capture the trend in a single video. So, over here, we visualize just the aggregate probability of a hexagonal zone being the destination of relocation (this is correlated with number of relocations to the zone). As expected, the relocation targets are primarily limited to downtown Manhattan and the 2 airports (regions where there is excess demand) across the day. 

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
