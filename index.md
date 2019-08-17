This is a webpage associated with a double-blind submission to WSDM 2020 conference. It contains a link to the codebase for the framework proposed in the paper along with a processed dataset required to reproduce its results.

Code - [Link to repository.](https://github.com/transparent-framework/optimize-ride-sharing-earnings)

Sample Data - [City states file.](https://drive.google.com/file/d/1908IwJPsp8DF9lHhObvBG693pRrb0GV6/view)

* Note: All the results presented in our paper, except generalizability of model result, use a representative day, i.e., first Monday of September 2015. The sample data file linked above - `city_states.dill` corresponds to this representative day. The experiments in our codebase can directly use this file as input to reproduce all the results, without having to pre-process the raw NYC Yellow taxi datasets.

* Reproducing the results of the generalizability experiment requires training of 210 separate models (7 models for each of the 30 days in the month) and testing the deployment of approximately approximately 2,000 models using over 30 GBs of data in form of city_states files for each day of the month. In order to keep hosting costs to a minimal, we recommend using our code to pre-process the raw-data from NYC Yellow Taxi dataset to those interested in reproducing this specific result.
