* Change working directory
cd "./data/"

* Load the data (Monday)
insheet using "rmb_data/rmb_data_Monday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Monday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Monday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Monday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------

* Load the data (Tuesday)
insheet using "rmb_data/rmb_data_Tuesday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Tuesday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Tuesday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Tuesday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------

* Load the data (Wednesday)
insheet using "rmb_data/rmb_data_Wednesday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Wednesday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Wednesday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Wednesday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------

* Load the data (Thursday)
insheet using "rmb_data/rmb_data_Thursday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Thursday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Thursday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Thursday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------

* Load the data (Friday)
insheet using "rmb_data/rmb_data_Friday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Friday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Friday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Friday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------

* Load the data (Saturday)
insheet using "rmb_data/rmb_data_Saturday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Saturday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Saturday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Saturday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* Load the data (Sunday)
insheet using "rmb_data/rmb_data_Sunday_september_15.csv", comma

* Fit model fare amount
eststo: quietly regress fare_amount i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Sunday_september_15_fare.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model trip distance
eststo: quietly regress trip_distance i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Sunday_september_15_trip.csv", replace wide plain noobs not nomtitles
eststo clear

* Fit model duration seconds
eststo: quietly regress duration_seconds i.pickup_hour c.straight_line_distance, noconstant

* Export the model in csv file
esttab using "rmb_models/Sunday_september_15_duration.csv", replace wide plain noobs not nomtitles
eststo clear
clear

* --------------------------
