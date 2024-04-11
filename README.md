# NYC Taxi Data Analysis
## Installation
If you want to run the report generation yourself, please run:
```sh
poetry install
```


## Seeing the interactive plots
You can find the interactive plots for monthly averages and 45 day rolling mean of the trip distance and trip duration in the `report` folder. The report is named `plots.html`.


## Recreating the plots
If you want to rerun the report generation, please run:
```sh
python create_report.py
```
**WARNING**: This will download the data automatically from [https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page?month=).
You can limit the data by passing e.g. the start year as an argument to the script, e.g.
```sh
python create_report.py --start-year 2023
```

## Observations

In the following, I will list some observations I made based on the plots I created:
* average trip distance decreases in the winter months, probably due to the cold weather, which makes people less likely to walk
* average trip duration increases in the winter months. This could have multiple reasons:
    * due to weather conditions (snow, ice) taxis have to drive slower
    * there is more car traffic in the winter months, because people are less likely to walk
    * shorter trips are more likely to have a lower average speed than longer trips since the time to board and off-board is a larger part of the total trip time
* Before 2014, the average trip distance and average time were both slightly increasing over time. 
* After 2014, the average trip distance is decreasing, while the average trip duration is increasing. This could be due to the increasing traffic in New York City, which makes it harder to get around, and therefore increases the time it takes to get from A to B.
* The pandemic is clearly visible as an anomaly in the data:
  * trip length in time spiking downwards around April 2020 which roughly coincides with the first lockdown in New York City. Potentially, trips were shorter in time because traffic was very low.
  * trip distance spiking upwards in June 2020, coinciding roughly with beginning of the reopening.
  * The matches are rough because the rolling average is taken over 45 days, so the effect lags a bit behind.
* After the pandemic, trip distance is a lot higher than before the pandemic. This is just speculation, but maybe this is an effect of more people working from home and less travelling for work because business people might be most inclined to use taxis for short distances since they usually get refunded for it.
* Average speed seems to be better after the pandemic, maybe also due to less traffic caused by fewer people commuting for work.
* What this data doesn't show us:
  * how and why people are using taxis has a big influence on average trip distance and duration. 
  * This is influenced a lot by the attractiveness of other modes of transport, e.g. public transport, biking, walking, e-scooters, other taxi-like modes of transport like e.g. Uber.