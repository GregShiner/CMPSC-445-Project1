# Philadelphia Housing Data Analysis

## Data Collection
The data for this project was gathered from the Philadelphia OPA Property Assesments dataset. The dataset was downloaded from the [OpenDataPhilly](https://www.opendataphilly.org/dataset/opa-property-assessments) website. The dataset contains information on property assessments in Philadelphia. The dataset contains 81 columns and 583,000 rows. The dataset was downloaded as a CSV file and was read into a pandas dataframe for analysis. I also considered scrapping data from the Zillow API, but the process for gaining access to the data was not as easy as just using the public housing data from OpenDataPhilly.

The data in this dataset contained primarily property inspection data for all types of properties in Philadelphia. The data included information on the property's inspection information, location, owner, taxes, and other property characteristics, such as number of bedrooms/bathrooms, and land size. 