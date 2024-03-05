# Philadelphia Housing Data Analysis

## Data Collection
The data for this project was gathered from the Philadelphia OPA Property Assesments dataset. The dataset was downloaded from the [OpenDataPhilly](https://www.opendataphilly.org/dataset/opa-property-assessments) website. The dataset contains information on property assessments in Philadelphia. The dataset contains 81 columns and 583,000 rows. The dataset was downloaded as a CSV file and was read into a pandas dataframe for analysis. I also considered scrapping data from the Zillow API, but the process for gaining access to the data was not as easy as just using the public housing data from OpenDataPhilly.

The data in this dataset contained primarily property inspection data for all types of properties in Philadelphia. The data included information on the property's inspection information, location, owner, taxes, and other property characteristics, such as number of bedrooms/bathrooms, and land size.

## Data Preprocessing
### Removing Columns With No Useful Information
The first step in cleaning up this data is removing columns that do not need to be considered. These columns could be removed for a variety of reasons but the most common one was that the data was simply not relevant to the analysis. There were many columns for information such as owner information, book keeping information, tax information, and some geographic information that was not relevant to the analysis. The columns that were kept were the columns that contained information on the property's characteristics, such as number of bedrooms, number of bathrooms, and land size. 

### Filter Categories
The next step in cleaning up this data was to filter the data to only include residential properties. This was done by filtering the data to only include properties that had a `category_code` of 1 or 2 which represented single family and multi-family homes respectively.

### Removing Columns With Too Much Missing Data
The next step in cleaning up this data was to remove columns that had too much missing data. The threshold for this was set at 25% missing data. This was done to ensure that the data was as clean as possible and that the analysis would not be skewed by missing data.

### Imputing Missing Data
The next step in cleaning up this data was to impute the missing data. The strategy for imputing the data varied for the different columns based on what type of data they were storing. For example, for columns that contained numerical data, the missing data was imputed with the median of the column. For categorical data, I created a new category called "unknown". An alternate option would have been to fill the missing data with the mode of the column, but I chose to use the median to avoid skewing the data. I also noticed that often times the missing data was actually significant in that it represented a value such as "Not Applicable" so in these cases, it doesn't make sense to fill the missing data with the mode of the column.

### Encoding Categorical Data
In this project, we have 2 types of categorical data:
1. One-Hot

    a. `category_code`

    b. `general_construction`

    c. `topography`

    d. `view_type`

    e. `zoning`

2. Binary
    
    d. `year_built_estimate`: This column contained the values "Y", "N", and empty. "Y" was mapped to `True` and everything else was mapped to `False`.

    b. `homestead_exemption`: This column is actually a numeric column, but the only 2 values are  80,000 and 0. 80,000 was mapped to `True` and 0 was mapped to `False`.

    c. `exempt_building`: Similarly, this column was mapped to `False` for values of 0, and `True` for all other columns.

    d. `exempt_land`: Once again, this column was mapped to `False` for values of 0, and `True` for all other columns.

### Removing outliers
The next step in cleaning up this data was to remove outliers. I actually opted to filter these outliers out by manually creating thresholds. I accomplished this by running the data through a data wrangling program like the Visual Studio Code Data Wrangler. I used the program to view the distributions of the numeric columns. I then used these graphs to spot outliers and then manually set thresholds for these columns. I then filtered the data to only include rows that were within these thresholds.