import sys; 

# %% 
import pandas as pd 
from sqlalchemy import create_engine 
import pyodbc as odbc 
import sqlalchemy 
import numpy as np 

# appear in Jupyter in notebooks 
import seaborn as sns 
import warnings # remove warnings 
import matplotlib.pyplot as plt 
import sklearn 
# warnings.filterwarnings("ignore", "is_categorical_dtype") 
# warnings.filterwarnings("ignore", "use_inf_as_na") 

# %% 
# Make a DataFrame 
engine = create_engine('mssql+pyodbc://localhost/DBA?driver=SQL+Server+Native+Client+11.0') 
growth_size = pd.read_sql('SELECT[DTINSERTED], dbserver, [DriveLetter],[Previous Space],[Current Space],[Growth],[TotalDriveSize],[PercentageFree],Case when [PercentageFree] < 30 then 1 when [PercentageFree] > 30 then 0 end as LowPercentFree FROM [dba].[dbo].[MonthGrowthSummary] with(nolock) order by Convert(char(10), [DTINSERTED],121) asc', 
                engine) 

# %% 
# function below will get the data in ready state to use the model to train the data. Feature engineering function to convert types 
def preprocess_data(growth_size): 
    growth_size["DTINSERTED"] = pd.to_datetime(growth_size["DTINSERTED"], errors="coerce") 
    """ 
    Performs transformations on growth_size and returnes transformed growth_size. 
    """ 
    # Preserve original text labels 
    growth_size["dbserver_text"] = growth_size["dbserver"] 
    growth_size["DriveLetter_text"] = growth_size["DriveLetter"] 

    # Extract date features 
    growth_size["DateFreeInAYear"] = growth_size.DTINSERTED.dt.year 
    growth_size["DateFreeInAMonth"] = growth_size.DTINSERTED.dt.month 
    growth_size["DateFreeInADay"] = growth_size.DTINSERTED.dt.day 
    growth_size["DateFreeInDayOfWeek"] = growth_size.DTINSERTED.dt.dayofweek 
    growth_size["DateFreeInDayOfYear"] = growth_size.DTINSERTED.dt.year 

    # fill the numeric rows with median 
    for label, content in growth_size.items(): 
        # Skip encoding for preserved text columns         
        if label in ["dbserver_text", "DriveLetter_text"]: 
            continue  # Skip encoding for preserved text columns 

        # This will turn all of string value into category values 
        if pd.api.types.is_string_dtype(content): 
            growth_size[label] = content.astype("category").cat.as_ordered() 
             
        if pd.api.types.is_numeric_dtype(content): 
            if pd.isnull(content).sum(): 
                #add binary column whih tells use if the data was missing 
                growth_size[label+"_is_missing"] =pd.isnull(content) 
                # fill missing numberic values with median 
                growth_size[label] = content.fillna(content.median()) 

        # filled categorical missing data and turn categories into numbers  
        if not pd.api.types.is_numeric_dtype(content): 
            # add binary column to indicate wheter sameple had missing vlue 
            growth_size[label+"_is_missing"] = pd.isnull(content) 
            #turn nubmer categories into number and add +1 
            growth_size[label] = pd.Categorical(content).codes+1 
             
    return growth_size 

# %% 
# Process the data from the pandas data frame 
growth_size_processed = preprocess_data(growth_size) 
#growth_size_processed.head() 

# %% 
growth_size_processed.head() 

# %% 
# Changing the datatype of Date, from 
# Object to datetime64ns 
# grabbing multiple columsn to make the date column and index on the date column preping for ploting 
growth_size_processed["date"] = pd.to_datetime(dict(year=growth_size_processed.DateFreeInDayOfYear, month=growth_size_processed.DateFreeInAMonth, day=growth_size_processed.DateFreeInADay))  
# Setting the Date as index 
dataframe = growth_size_processed.set_index("date") 

# Save text labels for later 
text_labels = growth_size_processed[["dbserver_text", "DriveLetter_text"]] 

#dataframe 

# %% 
# Import the RandomForestRegressor model class from the ensemble module 
from sklearn.ensemble import RandomForestRegressor 

# Setup random seed 
np.random.seed(42) 

# Split the data into X (features/data) and y (target/labels) 
#tempX = growth_size_processed.drop(["PercentageFree", "DTINSERTED_is_missing", "dbserver_is_missing", "DriveLetter_is_missing"], axis=1) 
tempX = growth_size_processed.drop(columns=["PercentageFree", "DTINSERTED_is_missing", "dbserver_is_missing", "DriveLetter_is_missing"]) 

X = tempX.set_index("date") 
y = growth_size_processed["PercentageFree"] 

# %% 
from sklearn.model_selection import train_test_split 

# Select only numeric columns for training 
X_numeric = X.select_dtypes(include=[np.number]) 

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42) 

# Train model 
rfr = RandomForestRegressor(n_estimators=100) 
rfr.fit(X_train, y_train) 

# Predict 
y_preds = rfr.predict(X_test) 

# %% 
#the next 3 code blocks will help determine how acurate the model is 
from sklearn.metrics import r2_score 

#Fill an array with y_test mean 
y_test_mean = np.full(len(y_test), y_test.mean()) 

# %% 
y_test_mean[:10] 

# %% 
r2_score(y_true=y_test, 
         y_pred=y_test_mean) 
# 0.0 

# %% 
r2_score (y_true=y_test, 
          y_pred=y_test) 
# this should give 1.0 

# %% 
from sklearn.metrics import mean_absolute_error 

mae = mean_absolute_error(y_test, y_preds) 
mae 

# %% 
df = pd.DataFrame(data={"actual values": y_test, 
                        "predicted values": y_preds}) 
df["differences"] = df["predicted values"] - df["actual values"] 
df.head() 

# %% 
# MAE using numpy functions 
np.abs(df["differences"]).mean() 

# %% 
r2_score (y_true=y_test, 
          y_pred=y_test) 

#end of model checks this should give 1.0 

# %% 
from sklearn.model_selection import cross_val_score 

cross_val_score = cross_val_score(rfr, X_numeric, y, cv=5) 

np.mean(cross_val_score) 

# %% 
import matplotlib.pyplot as plt 
#from statsmodels.graphics.regressionplots import abline_plot 
# Using a inbuilt style to change  
# the look and feel of the plot 
plt.style.use("fivethirtyeight") 

# setting figure size to 12, 10 
plt.figure(figsize=(20, 10)) 
  
# Labelling the axes and setting a  
# title 
plt.xlabel("Date") 
plt.ylabel("Percent available") 
plt.title("Mean percent available MSSQL", fontsize=15, fontweight='bold') 
sns.lineplot(x=X_test.index, 
             y=y_test, #dataframe["PercentageFree"], 
             #color='0.75', 
             label='% Free') 

sns.lineplot(x=X_test.index, 
             y=y_preds, 
           # color='b', 
             linestyle='--', 
             label='Predicted % free') 

plt.xticks(rotation=45, fontsize=11) 

plt.axhline(y_test.mean(),linestyle='-', label=f"Mean % Available {y_test.mean()}") 

plt.legend() 
plt.savefig(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\MeanpercentavailableFig.png"); 
#plt.show() 

# %% 
import matplotlib.pyplot as plt 
#from statsmodels.graphics.regressionplots import abline_plot 
# Using a inbuilt style to change  
# the look and feel of the plot 
plt.style.use("fivethirtyeight") 

# setting figure size to 12, 10 
plt.figure(figsize=(20, 10)) 
  
# Labelling the axes and setting a  
# title 
plt.xlabel("Date") 
plt.ylabel("Percent available") 
plt.title("Mean percent available MSSQL", fontsize=15, fontweight='bold') 

plt.scatter(x=X_test.index, 
            y=y_test, 
            color='c', 
            label='Actual % free') 

plt.scatter(x=X_test.index, 
            y=y_preds, 
            color='r', 
            label='Predicted % free') 

plt.xticks(rotation=45, fontsize=11) 

#plt.axhline(y_test.mean(),linestyle='-', color='k',label=f"Mean percent available {y_test.mean()}") 
plt.axhline(y_test.mean(),linestyle='-', label=f"Mean % Available {y_test.mean()}") 
#plt.axhline(y_preds.mean(),linestyle='-', color='w',label=f"Mean percent predicted {y_preds.mean()}") 

# plt.axhline(y_preds.mean(), 
#           linestyle='--', color='r',label=f"Predicted {y_preds.mean()}") 
plt.legend() 
plt.savefig(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\Meanpercentavailablescatter.png"); 
#plt.show() 

# %% 
# New code from AI 

#Option 1 

from sklearn.linear_model import LinearRegression 

# Prepare a DataFrame to store forecasts 
forecast_results = [] 

# Loop through each dbserver 
for server in growth_size_processed["dbserver"].unique(): 
    # Filter data for this server 
    server_data = growth_size_processed[growth_size_processed["dbserver"] == server] 
     
    # Sort by date 
    server_data = server_data.sort_values("date") 
     
    # Convert date to ordinal for regression 
    X = server_data["date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1) 
    y = server_data["PercentageFree"].values 
     
    # Fit linear regression 
    model = LinearRegression() 
    model.fit(X, y) 
     
    # Forecast next 30 days 
    future_dates = pd.date_range(start=server_data["date"].max() + pd.Timedelta(days=1), periods=30) 
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1) 
    future_preds = model.predict(future_ordinals) 
     
    # Store results 
    forecast_df = pd.DataFrame({ 
        "dbserver": server, 
        "date": future_dates, 
         
        "predicted_PercentageFree": future_preds 
    }) 
    forecast_results.append(forecast_df) 

# Combine all forecasts 
forecast_all = pd.concat(forecast_results) 
print(forecast_all.head()) 

# %% 

from sklearn.linear_model import LinearRegression 

forecast_results = [] 

# Define forecast horizons 
horizons = [30, 60, 90, 365] 

# Loop through each dbserver and drive letter combination 
for (server_code, drive_code), group in growth_size_processed.groupby(["dbserver", "DriveLetter"]): 
    group = group.sort_values("date") 

    # Convert date to ordinal 
    X = group["date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1) 
    y = group["PercentageFree"].values 

    # Fit model 
    model = LinearRegression() 
    model.fit(X, y) 

    # Forecast for each horizon 
    for horizon in horizons: 
        future_dates = pd.date_range(start=group["date"].max() + pd.Timedelta(days=1), periods=horizon) 
        future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1) 
        future_preds = model.predict(future_ordinals) 

        # Use original text labels from the group 
        server_name = group["dbserver_text"].iloc[0] 
        drive_letter = group["DriveLetter_text"].iloc[0] 

        forecast_df = pd.DataFrame({ 
            "dbserver": server_name, 
            "DriveLetter": drive_letter, 
            "date": future_dates, 
            "predicted_PercentageFree": future_preds, 
            "forecast_horizon": f"{horizon}_days" 
        }) 

        forecast_results.append(forecast_df) 

# Combine all forecasts 
forecast_all = pd.concat(forecast_results) 

# Show sample output 
print(forecast_all.head(20)) 

# Grouping forecasts by dbserver, drive letter, and forecast horizon 
grouped_forecast = forecast_all.groupby(["dbserver", "DriveLetter", "forecast_horizon"])["predicted_PercentageFree"].mean().reset_index() 
print(grouped_forecast) 

# %% 

# Export to CSV 
grouped_forecast.to_csv(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\grouped_forecast.csv", index=False) 

# %% 

import pandas as pd 
import plotly.express as px 

# Load the full forecast data 
forecast_all = pd.read_csv(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\grouped_forecast.csv") 

# Filter rows where predicted free space is less than 30% 
filtered_forecast = forecast_all[forecast_all["predicted_PercentageFree"] < 30].copy() 

# Create a new column combining dbserver and DriveLetter for labeling 
filtered_forecast.loc["DriveLetter"] = filtered_forecast["dbserver"] + " - Drive " + filtered_forecast["DriveLetter"] 

# Set custom order for forecast_horizon 
filtered_forecast["forecast_horizon"] = pd.Categorical( 
    filtered_forecast["forecast_horizon"], 
    categories=["30_days", "60_days", "90_days", "365_days"], 
    ordered=True 
) 

# Sort by forecast_horizon and then by predicted percentage 
filtered_forecast = filtered_forecast.sort_values(["forecast_horizon", "predicted_PercentageFree"]) 

# Create a line plot grouped by forecast horizon 
fig = px.bar( 
    filtered_forecast, 
    x="forecast_horizon", 
    y="predicted_PercentageFree", 
    color="dbserver", 
    barmode="group", 
    title="Forecasted Drive Space < 30% Free by Server and Drive", 
    labels={ 
        "predicted_PercentageFree": "% Free", 
        "dbserver": "Server & Drive", 
        "forecast_horizon": "Forecast Horizon"     
    } 
) 
# Export filtered data to CSV 
filtered_forecast.to_csv(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\forecast_below_30_labeled.csv", index=False) 
fig.write_image(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\forecast_below_30_labeled.png") 

# Save the plot 
#fig.write_image(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\forecast_below_30_labeled.png") 
#fig.write_json(r"C:\Users\<id?>\Desktop\timeseries\matplotlib\env\jupyternotebook\upload\forecast_below_30_labeled.json") 
# %%