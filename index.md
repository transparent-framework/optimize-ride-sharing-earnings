This is a webpage associated with a double-blind submission to WSDM 2020 conference. It contains a link to the codebase for the framework proposed in the paper along with a processed dataset required to reproduce its results.

Code - [Link to the repository.](https://github.com/transparent-framework/optimize-ride-sharing-earnings)

Sample Data - [City states file required to reproduce results presented in paper.](https://drive.google.com/file/d/1908IwJPsp8DF9lHhObvBG693pRrb0GV6/view)

* Note: All results in the paper, except generalizability of model result, are illustrated using a representative day, i.e., first Monday of September 2015. The sample data `city_states.dill` corresponds to this representative day. The experiments in the code-base can directly use this file as input to reproduce all the results, without having to pre-process raw NYC Yello taxi datasets.

* Reproducing the results of the generalizability experiment requires training of 210 separate models and testing of approximately approximately 2,000 models using 30 GBs of data. In order to reproduce this result, we recommend using our code to pre-process the raw-data from NYC Yellow Taxi dataset.
