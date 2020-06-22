This is a webpage associated with the paper "[Learn to Earn: Enabling Coordination Within a Ride-Hailing Fleet](http://arxiv.org/abs/2006.10904)". It contains a link to the codebase for the framework proposed in the paper along with a processed dataset required to reproduce its results.

Code - [Link to repository.](https://github.com/transparent-framework/optimize-ride-sharing-earnings)

Sample Data - [City states file.](https://drive.google.com/file/d/1908IwJPsp8DF9lHhObvBG693pRrb0GV6/view)

* Note: All the results presented in our paper, except generalizability of model result, use a representative day, i.e., first Monday of September 2015. The sample data file linked above - `city_states.dill` corresponds to this representative day. The experiments in our codebase can directly use this file as input to reproduce all the results, without having to pre-process the raw NYC Yellow taxi datasets.

* Reproducing the results of the generalizability experiment requires training of 210 separate models (7 models for each of the 30 days in the month) and testing the deployment of approximately approximately 2,000 models using over 30 GBs of data in form of city_states files for each day of the month. In order to keep hosting costs to a minimal, we recommend using our code to pre-process the raw-data from NYC Yellow Taxi dataset to those interested in reproducing this specific result.

### Visualization videos

#### Degree of coordination evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/coordination_probability.mp4?raw=true)
In this framework, for lack of a better alternative, at the start of the day all the drivers are uniformly distributed across
the greater New York City. As a result, degree of coordination is high all across the city (except Manhattan which still has demand at midnight) facilitating the movement of drivers from all across the city to neighborhoods with the demand. As the day progresses, we observe that coordination is required in varying degrees in 3 major spots viz., downtown Manhattan and the two airports. We see a smooth increase/decrease of coordination over geographical region and time, thereby confirming that our framework is learning supply-demand characteristics as they evolve.

#### Coordinated wait action evolution: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/wait_probability.mp4?raw=true)
Based on the need of coordination, in this video we show the probability that the recommended coordination action to the drivers is waiting in their current location. Interestingly, we find that probability of coordinated wait action is distributed
all across the greater NYC in the morning rush hours (movement towards Manhattan from outside boroughs). As the day progresses, the coordinated wait is limited to Manhattan and sometimes at the 2 airports. As the evening approaches, we see that the region for coordinated wait action gradually expands to cover neighborhoods outside downtown Manhattan.

#### Popular relocation zones during coordination: [mp4 file](https://github.com/transparent-framework/optimize-ride-sharing-earnings/blob/master/data/relocation_probability.mp4?raw=true)
Based on the need of coordination, in this video we attempt to show the popularity of relocation targets. As each relocation ride has a source and destination, it is difficult to capture the trend in a single video. So, over here, we visualize just the aggregate probability of a hexagonal zone being the destination of relocation (this is correlated with number of relocations to the zone). As expected, the relocation targets are primarily limited to downtown Manhattan and the 2 airports (regions where there is excess demand) across the day. 
