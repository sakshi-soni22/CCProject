# %%
pip install yfinance

# %%
import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import date

# %%
tickers = ['MSFT', 'GOOG']

# %%
#Fetch last 5 years data of Apple and Microsoft
data = yf.download(tickers, start='2018-01-01', end=date.today())

# %%
#Display first few rows
print(data.head())

# %%
#Display last few rows
print(data.tail())

# %%
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import numpy as np
from datetime import date

# %%
ticker = 'MSFT'
start_date = '2001-01-01'
end_date = date.today()

# Fetch the historical data
data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print(data.head())

# %% [markdown]
# ADJUSTED CLOSING PRICE- Price after accounting for events like dividends and ticker splits.

# %%
#Plot the Adjusted Closing Price
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Adjusted Close', color='blue')
plt.title('Adjusted Closing Prices of MSFT')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# HISTOGRAM AND DENSITY CURVE FOR ADJUSTED CLOSING PRICE
# 
# x-axis represents the range of adjusted closing prices, and the y-axis shows how many times the ticker closed at each price or within each price range.
# This helps to understand the overall price movement, where the ticker typically closed, and if there were any frequent price points (peaks) or large price changes 

# %%
plt.figure(figsize=(14, 7))
sns.histplot(data['Close'], kde=True)
plt.title('Histogram and Density Plot of Adjusted Closing Prices')
plt.xlabel('Adjusted Closing Price')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# BOX PLOT AND VIOLIN PLOT:
# 
# Box Plot shows-
# 
# Focuses on the key statistics (median, quartiles, and outliers) of the adjusted closing price distribution.
# 
# Violin Plot shows- 
# 
# Shows both the spread and the shape of the data, along with the central tendency and potential outliers. The distribution shape (whether the ticker’s adjusted closing price is skewed, bimodal, etc.).

# %%
plt.figure(figsize=(14, 7))

# Box Plot
plt.subplot(121)
sns.boxplot(data['Close'])
plt.title('Box Plot of Adjusted Closing Prices')

# Violin Plot
plt.subplot(122)
sns.violinplot(data['Close'])
plt.title('Violin Plot of Adjusted Closing Prices')

plt.show()

# %% [markdown]
# Taking log of the prices helps to normalize the data, remove outliers and reduce skewness, making it easier to observe patterns and draw insights from the distribution.

# %%
plt.figure(figsize=(14, 7))

# Box Plot
plt.subplot(121)
sns.boxplot(np.log(data['Close']))
plt.title('Box Plot of Adjusted Closing Prices')

# Violin Plot
plt.subplot(122)
sns.violinplot(np.log(data['Close']))
plt.title('Violin Plot of Adjusted Closing Prices')

plt.show()

# %% [markdown]
# MEAN, VARIANCE, STANDARD DEVIATION, SKEWNESS, KURTOSIS
# 
# Mean: The average value of a dataset, calculated by summing all values and dividing by the number of values.
# 
# Variance: A measure of how much the data points differ from the mean, representing the spread of the data.
# 
# Standard Deviation: The square root of the variance, indicating how much data points deviate from the mean on average.
# 
# Skewness: A measure of the asymmetry or lopsidedness of the data distribution.
# (>0 right skewed, <0 left skewed, approximately 0 symmetric)
# 
# Kurtosis: A measure of the "tailedness" or sharpness of the peak in the data distribution, indicating the presence of outliers.

# %%
mean = data['Close'].mean()
variance = data['Close'].var()
std_dev = data['Close'].std()
skewness = data['Close'].skew()
kurtosis = data['Close'].kurt()

#print(f"Mean: {mean}, Variance: {variance}, Std Dev: {std_dev}, Skewness: {skewness}, Kurtosis: {kurtosis}")
print(f"Mean: {mean.values[0]}")
print(f"Variance: {variance.iloc[0]}")
print(f"Standard Deviation: {std_dev.iloc[0]}")
print(f"Skewness: {skewness.iloc[0]}")
print(f"Kurtosis: {kurtosis.iloc[0]}")

# %% [markdown]
# MISSING VALUES- Handling missing values using forward fill
# 

# %%
missing_values = data.isnull().sum()
print("Missing values before handling:\n", missing_values)  #no missing values here

data_ffill = data.ffill()

missing_values_after_ffill = data_ffill.isnull().sum()
print("Missing values after forward fill:\n", missing_values_after_ffill)


# %% [markdown]
# ## AUTOCORRELATION AND PARTIAL AUTOCORRELATION:
# 
# Autocorrelation- measures the relationship between a time series and its own past values (lags). It includes the impact of all previous lags.
# 
# Partial Autocorrelation-  Focuses only on the direct relationship with a specific lag, ignoring indirect influences from intermediate lags.

# %%
plt.figure(figsize=(14, 7))
plt.subplot(211)
plot_acf(data_ffill['Close'].dropna(), lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(data_ffill['Close'].dropna(), lags=50, ax=plt.gca())
plt.show()

# %% [markdown]
# The autocorrelation gradually decreases with lags, indicating that the series likely has a trend or is non-stationary.
# 
# There is a strong positive partial autocorrelation at lag 1, indicating that the immediate past value directly influences the current value.
# Drop After Lag 1:
# The PACF drops quickly after lag 1 and stabilizes around zero for higher lags. This suggests that only the immediate lag is important, and the influence of other lags is mediated through the first lag.
# 

# %% [markdown]
# DESCRIPTIVE STATISTICS SUMMARY

# %%
desc_stats = data.describe()
print(desc_stats)

# %%
print(f"Skewness: {skewness.iloc[0]}")
print(f"Kurtosis: {kurtosis.iloc[0]}")

# %%
# Fetch MSFT and GOOG data from Yahoo Finance
ticker_msft = "MSFT"
ticker_goog = "GOOG"

# Download the data
data_msft = yf.download(ticker_msft, start="2005-08-12", end=date.today())
data_goog = yf.download(ticker_goog, start="2005-08-12", end=date.today())
data_msft.columns = data_msft.columns.get_level_values(0)
data_goog.columns = data_goog.columns.get_level_values(0)

# Display the first few rows of each dataset to confirm successful download
print(data_msft.head())
print(data_goog.head())
data_msft.dropna(inplace=True)
data_goog.dropna(inplace=True)

# %% [markdown]
# MSFT and GOOG CLOSING PRICES OVER TIME:

# %%
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close'], label='Microsoft Close Prices', color='blue')
plt.plot(data_goog['Close'], label='Google Close Prices', color='red')
plt.title('Microsoft vs Google Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# %% [markdown]
# ## COVARIANCE, PEARSON CORRELATION AND SPEARMAN CORRELATION
# 
# Covariance: Measures how two variables vary together. (+ve variables increases together, -ve as one increases other decreases, 0 no relation)
# 
# Pearson Correlation: Measures the linear relationship between two variables. (normally distributed data)
# 
# Spearman Correlation: Measures the monotonic relationship between two variables(linear or non-linear). If one increases, the other consistently in or decreases.

# %%
#Covariance
cov_matrix = np.cov(data_msft['Close'], data_goog['Close'])
covariance = cov_matrix[0, 1]
print(f"Covariance between MSFT and GOOG: {covariance:.2f}")

# %%
#Pearson Correlation
pearson_corr = data_msft['Close'].corr(data_goog['Close'], method='pearson')
print(f"Pearson Correlation between MSFT and GOOG: {pearson_corr:.2f}")


# %%
#Spearman Correlation 
spearman_corr = data_msft['Close'].corr(data_goog['Close'], method='spearman')
print(f"Spearman Correlation between MSFT and GOOG: {spearman_corr:.2f}")

# %% [markdown]
# SCATTER PLOT: help identify patterns, trends, and correlations between two variables.
# 
# If the points are closely clustered around an upward-sloping line, it indicates a strong positive correlation.
# 
# A scattered or random distribution of points suggests little to no correlation, indicating the two stocks move independently of each other.
# 
# If points cluster around a downward-sloping line, it suggests a negative correlation

# %%
plt.figure(figsize=(10, 6))
plt.scatter(data_msft['Close'], data_goog['Close'], alpha=0.5, color='purple')
plt.title('Scatter Plot of MSFT vs GOOG Close Prices')
plt.xlabel('MSFT Close Price (INR)')
plt.ylabel('GOOG Close Price (INR)')
plt.show()


# %% [markdown]
# ## STATIONARITY OF DATA
# 
# - Stationarity is essential for accurate and reliable time series forecasting. - Non-stationary data can lead to poor model performance and inaccurate predictions.
# - ADF Test is used to check for stationarity. A significant p-value indicates that the series is stationary.
# 
# Techniques to Achieve Stationarity:
# - Differencing removes trends.
# - Log Transformation stabilizes variance.
# - De-Trending removes long-term trends.
# - De-Seasonalizing removes seasonal patterns.

# %% [markdown]
# 1. ADF (Augmented Dickey-Fuller test) 
# 
# Null Hypothesis (H₀): The time series has a unit root (non-stationary).
# Alternative Hypothesis (H₁): The time series is stationary.
# 
# p-value < 0.05: Reject H₀, meaning the series is stationary.
# 
# p-value ≥ 0.05: Fail to reject H₀ meaning the series is non-stationary.

# %%
from statsmodels.tsa.stattools import adfuller

# Perform ADF test on MSFT Close prices
adf_result_msft = adfuller(data_msft['Close'])

# Extracting test statistics
adf_statistic_msft = adf_result_msft[0]
p_value_msft = adf_result_msft[1]
critical_values_msft = adf_result_msft[4]

print(f"ADF Statistic: {adf_statistic_msft:.4f}")
print(f"P-Value: {p_value_msft:.4f}")
print("Critical Values:")
for key, value in critical_values_msft.items():
    print(f"{key}: {value:.4f}")

# Interpretation
if p_value_msft < 0.05:
    print("Reject the null hypothesis: The series is stationary.")
else:
    print("Fail to reject the null hypothesis: The series is non-stationary.")


# %% [markdown]
# 2. First Order Differencing

# %%
# Apply first-order differencing to MSFT Close prices
data_msft['Close_diff'] = data_msft['Close'].diff().dropna()

# Re-run the ADF test on differenced data
adf_result_diff_msft = adfuller(data_msft['Close_diff'].dropna())

print(f"ADF Statistic (Differenced Data - MSFT): {adf_result_diff_msft[0]:.4f}")
print(f"P-Value (Differenced Data - MSFT): {adf_result_diff_msft[1]:.4f}")

# Plot the differenced data
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close_diff'], label='Differenced MSFT Close Prices')
plt.title('Differenced MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Price Difference (INR)')
plt.legend()
plt.show()


# %% [markdown]
# 3. Log Transformation

# %%
# Apply log transformation to MSFT Close prices
data_msft['Close_log'] = np.log(data_msft['Close'])

# Re-run the ADF test on log-transformed data
adf_result_log_msft = adfuller(data_msft['Close_log'].dropna())

print(f"ADF Statistic (Log Transformed Data - MSFT): {adf_result_log_msft[0]:.4f}")
print(f"P-Value (Log Transformed Data - MSFT): {adf_result_log_msft[1]:.4f}")

# Plot the log-transformed data
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close_log'], label='Log Transformed MSFT Close Prices', color='purple')
plt.title('Log Transformed MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Log(Price)')
plt.legend()
plt.show()


# %% [markdown]
# 4. DE-TRENDING DATA BY SUBTRACTING ROLLING MEAN
# 
# A rolling mean (also called a moving average) is a statistical technique used to smooth out short-term fluctuations and highlight longer-term trends in a dataset, particularly in time series data. The rolling mean is calculated by taking the average of a fixed window

# %%
# De-trend by subtracting the rolling mean
rolling_mean = data_msft['Close'].rolling(window=10).mean()
data_msft['Close_detrended'] = data_msft['Close'] - rolling_mean

# Re-run the ADF test on differenced data
adf_result_detrended = adfuller(data_msft['Close_detrended'].dropna())

print(f"ADF Statistic (Differenced Data): {adf_result_detrended[0]:.4f}")
print(f"P-Value (Differenced Data): {adf_result_detrended[1]:.4f}")

# Plot the de-trended data
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close_detrended'], label='De-Trended Close Prices', color='green')
plt.title('De-Trended MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# %% [markdown]
# 5. DE-SEASONALIZING DATA
# 
# Deseasonalizing data using seasonal decomposition involves separating a time series into its key components (trend, seasonality, and residuals) to remove the seasonal effects.

# %%
# De-seasonalizing using seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
decomposition_msft = seasonal_decompose(data_msft['Close'], model='additive', period=250)

# Extract and plot the seasonal component
seasonal_msft = decomposition_msft.seasonal
plt.figure(figsize=(14, 7))
plt.plot(seasonal_msft, label='Seasonal Component (MSFT)', color='orange')
plt.title('Seasonal Component of MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Seasonal Effect')
plt.legend()
plt.show()

# De-seasonalized data (original - seasonal component)
data_msft['Close_deseasonalized'] = data_msft['Close'] - seasonal_msft

# Re-run the ADF test on de-seasonalized data
adf_result_deseasonalized_msft = adfuller(data_msft['Close_deseasonalized'].dropna())

print(f"ADF Statistic (De-seasonalized Data - MSFT): {adf_result_deseasonalized_msft[0]:.4f}")
print(f"P-Value (De-seasonalized Data - MSFT): {adf_result_deseasonalized_msft[1]:.4f}")


# %% [markdown]
# - First Differencing: Often enough to remove trends and achieve stationarity, as confirmed by both the ADF and KPSS tests.
# - Second Differencing: Applied if first differencing is insufficient; should achieve stationarity in most cases.
# - Detrending: Removes the trend component, making the series stationary if the trend is the primary cause of non-stationarity.
# - Logarithmic Transformation: Stabilizes the variance, useful for data exhibiting exponential growth or increasing variance over time.

# %% [markdown]
# ## CHARACTERISTICS OF TIME SERIES DATA:

# %% [markdown]
# 1. VOLATILITY CLUSTERING- Periods of high volatility i.e. high fluctuations

# %%
# Calculate daily returns
data_msft['Returns'] = np.log(1+data_msft['Close'].pct_change())

# Plot the absolute returns to visualize volatility clustering
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Returns'].abs(), color='orange', label='Absolute Returns')
plt.title('Volatility Clustering in MSFT Returns')
plt.xlabel('Date')
plt.ylabel('Absolute Returns')
plt.legend()
plt.show()


# %% [markdown]
# 2. NOISE- Filtering out noise (unwanted data) is essential for building accurate forecasting models.

# %%
# Smooth the time series using a rolling mean
data_msft['Smoothed_Close'] = data_msft['Close'].rolling(window=50).mean()

# Plot the original and smoothed series
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close'], label='Original Close Prices', color='blue', alpha=0.5)
plt.plot(data_msft['Smoothed_Close'], label='Smoothed Close Prices (30-Day MA)', color='red')
plt.title('Original vs Smoothed MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()


# %% [markdown]
# 3. FAT TAILS- Distributions of returns with fatter tails than a normal distribution, indicating a higher likelihood of extreme events.
# 
# Indicates high kurtosis

# %%
import scipy.stats as stats

# Plot the distribution of MSFT returns
plt.figure(figsize=(10, 6))
sns.histplot(data_msft['Returns'].dropna(), bins=100, kde=True, color='purple', stat="density")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, data_msft['Returns'].mean(), data_msft['Returns'].std())
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.title('Distribution of MSFT Returns with Normal Distribution Overlay')
plt.xlabel('Returns')
plt.ylabel('Density')
plt.legend()
plt.show()

# %% [markdown]
# 4. AUTOCORRELATION- The correlation of a time series with a lagged version of itself

# %%
from statsmodels.graphics.tsaplots import plot_acf

# Plot ACF for MSFT Returns
plt.figure(figsize=(12, 6))
plot_acf(data_msft['Returns'].dropna(), lags=50, alpha=0.05)
plt.title('Autocorrelation Function (ACF) for MSFT Returns')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()

# Plot ACF for MSFT Price
plt.figure(figsize=(12, 6))
plot_acf(data_msft['Close'].dropna(), lags=50, alpha=0.05)
plt.title('Autocorrelation Function (ACF) for MSFT Closing Prices')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# %% [markdown]
# 5. MEAN REVERSION

# %%
plt.figure(figsize=(14, 7))
plt.plot(data_msft['Close'], label='MSFT Close Prices', color='blue')
plt.plot(rolling_mean, label='Rolling Mean (1-Year)', color='red')
plt.title('MSFT Close Prices and Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# %% [markdown]
# 6. NON-STATIONARITY (Done above)- Financial time series often exhibit non-stationarity due to trends, seasonality, or structural breaks. These characteristics must be addressed before applying certain forecasting models.

# %% [markdown]
# ## DATA PRE-PROCESSING TECHINQUES

# %% [markdown]
# 1. REMOVING MISSING VALUES- Essential to avoid biases in the model. Techniques include deletion, imputation, and interpolation.

# %%
missing_data = data_msft.isnull().sum()
print("Missing values in each column:\n", missing_data)

# %%
# Drop rows with missing values
data_msft_cleaned = data_msft.dropna()

print("Missing values after deletion:\n", data_msft_cleaned.isnull().sum())

# %%
# Fill missing values using forward fill method
data_msft_ffill = data_msft.fillna(method='ffill')

# Fill missing values using backward fill method as a fallback
data_msft_bfill = data_msft_ffill.fillna(method='bfill')

# Fill missing values using interpolation
data_msft_interpolated = data_msft_bfill.interpolate(method='linear')

# Choose the method and finalize the dataset
data_msft_final = data_msft_interpolated.copy()

# Confirm missing values are handled
print("Missing values after imputation:\n", data_msft_final.isnull().sum())

# %%
data_msft_final

# %% [markdown]
# OUTLIERS USING Z-SCORE

# %%
from scipy import stats

# Calculate Z-scores
data_msft_final['Z-Score'] = np.abs(stats.zscore(data_msft_final['Close']))

# Define a threshold to identify outliers (e.g., Z-score > 3)
outliers_z = data_msft_final[data_msft_final['Z-Score'] > 3]
print(f"Number of outliers detected by Z-Score: {len(outliers_z)}")

# Plot to visualize outliers
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.scatter(outliers_z.index, outliers_z['Close'], color='red', label='Outliers (Z-Score > 3)')
plt.title('Outlier Detection using Z-Score')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# %% [markdown]
# OUTLIERS USING INTERQUARTILE RANGE

# %%
# Calculate IQR
Q1 = data_msft_final['Close'].quantile(0.25)
Q3 = data_msft_final['Close'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers based on IQR
outliers_iqr = data_msft_final[(data_msft_final['Close'] < (Q1 - 1.5 * IQR)) | (data_msft_final['Close'] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers detected by IQR method: {len(outliers_iqr)}")

# Plot to visualize outliers
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.scatter(outliers_iqr.index, outliers_iqr['Close'], color='red', label='Outliers (IQR Method)')
plt.title('Outlier Detection using IQR Method')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()

# %% [markdown]
# 2. REMOVAL OF OUTLIERS

# %%
# Option 1: Remove outliers 
data_MSFT_outliers_removed = data_msft_final[(data_msft_final['Close'] >= (Q1 - 1.5 * IQR)) & (data_msft_final['Close'] <= (Q3 + 1.5 * IQR))]

# Option 2: Cap outliers
capped_values = data_msft_final['Close'].clip(lower=(Q1 - 1.5 * IQR), upper=(Q3 + 1.5 * IQR))

# Plot to compare the original vs. treated data
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='Original Close Prices', color='blue', alpha=0.5)
plt.plot(capped_values, label='Capped Close Prices', color='red')
plt.title('Comparison of Original and Treated Data (Outlier Capping)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()


# %% [markdown]
# 3. NORMALIZATION- Key for rescaling data, especially when dealing with features of different scales or units.

# %%
from sklearn.preprocessing import MinMaxScaler

# Apply Min-Max Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data_MSFT_normalized = scaler.fit_transform(data_msft_final[['Close']])

# Plot the normalized data
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final.index, data_MSFT_normalized, label='Normalized Close Prices', color='purple')
plt.title('Normalized MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()


# %% [markdown]
# 4. STANDERDIZATION- Key for rescaling data, especially when dealing with features of different scales or units.

# %%
from sklearn.preprocessing import StandardScaler

# Apply Standardization
scaler = StandardScaler()
data_MSFT_standardized = scaler.fit_transform(data_msft_final[['Close']])

# Plot the standardized data
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final.index, data_MSFT_standardized, label='Standardized Close Prices', color='orange')
plt.title('Standardized MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Standardized Price')
plt.legend()
plt.show()

# %% [markdown]
# 5. LAGGED VARIABLES CREATION: Critical for capturing temporal dependencies in time series forecasting.

# %%
# Create lagged features (e.g., 1-day, 2-day, and 3-day lags)
data_msft_final['Lag_1'] = data_msft_final['Close'].shift(1)
data_msft_final['Lag_2'] = data_msft_final['Close'].shift(2)
data_msft_final['Lag_3'] = data_msft_final['Close'].shift(3)

# Drop NA values generated by shifting
data_msft_final = data_msft_final.dropna()

# Display the first few rows to show lagged variables
print(data_msft_final[['Close', 'Lag_1', 'Lag_2', 'Lag_3']].head())

# %% [markdown]
# 6. REMOVE TRENDS
# 
# Differencing: A technique used to achieve stationarity by removing trends or seasonality in time series data.

# %%
# Apply differencing to remove trends
data_msft_final['Differenced'] = data_msft_final['Close'].diff()

# Plot the differenced data
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final.index, data_msft_final['Differenced'], label='Differenced Close Prices', color='red')
plt.title('Differenced MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.legend()
plt.show()


# %% [markdown]
# ROLLING MEAN AND ROLLING STANDARD DEVIATION

# %%
# Calculate rolling mean and standard deviation (e.g., 50-day rolling window)
rolling_mean_50 = data_msft_final['Close'].rolling(window=50).mean()
rolling_std_50 = data_msft_final['Close'].rolling(window=50).std()

# Plot the rolling mean and standard deviation
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(rolling_mean_50, label='50-Day Rolling Mean', color='red')
plt.plot(rolling_std_50, label='50-Day Rolling Std Dev', color='green')
plt.title('MSFT Close Prices with 50-Day Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()


# %% [markdown]
# ROLLING VARIANCE

# %%
rolling_variance = data_msft_final['Close'].rolling(window=50).var()
plt.figure(figsize=(14, 7))
plt.plot(rolling_variance, label='50-Day Rolling Variance', color='green')
plt.title('50-Day Rolling Variance of MSFT Close Prices')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# TIME SERIES DECOMPOSITION

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform additive decomposition
result_additive = seasonal_decompose(data_msft_final['Close'], model='additive', period=365)

# Plot the decomposition results
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(result_additive.observed, label='Observed', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(result_additive.trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(result_additive.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(result_additive.resid, label='Residual', color='orange')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# %%
# Perform multiplicative decomposition
result_multiplicative = seasonal_decompose(data_msft_final['Close'], model='multiplicative', period=365)

# Plot the decomposition results
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(result_multiplicative.observed, label='Observed', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(result_multiplicative.trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(result_multiplicative.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(result_multiplicative.resid, label='Residual', color='orange')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# %% [markdown]
# - Additive Model: Suitable for data where seasonal variations are constant over time. The observed series is the sum of the trend, seasonal, and residual components.
# - Multiplicative Model: Suitable for data where seasonal variations change in amplitude over time. The observed series is the product of the trend, seasonal, and residual components.
# - Purpose: Decomposition helps in isolating the individual components of a time series, leading to better insights and more accurate forecasting.

# %% [markdown]
# ## MOVING AVERAGES AND SMOOTHENING TECHNIQUES

# %% [markdown]
# SIMPLE MOVING AVERAGE- A basic technique for smoothing data to identify trends, using a fixed window of data points.

# %%
# Calculate 50-day Simple Moving Average (SMA)
sma_50 = data_msft_final['Close'].rolling(window=50).mean()

# Plot the SMA
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(sma_50, label='50-Day SMA', color='red')
plt.title('MSFT Close Prices with 50-Day Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# EXPONENTIAL MOVING AVERAGE- More responsive to recent changes, useful in scenarios where the latest data is more relevant.

# %%
# Calculate 50-day Exponential Moving Average (EMA)
ema_50 = data_msft_final['Close'].ewm(span=50, adjust=False).mean()

# Plot the EMA
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(ema_50, label='50-Day EMA', color='green')
plt.title('MSFT Close Prices with 50-Day Exponential Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# WEIGHTED MOVING AVERAGE- Assigns different levels of importance to data points, emphasizing more recent observations.

# %%
# Function to calculate Weighted Moving Average
def weighted_moving_average(data, window):
    weights = np.arange(1, window + 1)
    return data.rolling(window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

# Calculate 50-day Weighted Moving Average (WMA)
wma_50 = weighted_moving_average(data_msft_final['Close'], window=50)

# Plot the WMA
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(wma_50, label='50-Day WMA', color='purple')
plt.title('MSFT Close Prices with 50-Day Weighted Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# LOESS SMOOTHENING- Non-parametric, locally weighted smoothing that helps uncover trends in non-linear data.

# %%
import statsmodels.api as sm

# Apply LOESS smoothing
lowess = sm.nonparametric.lowess(data_msft_final['Close'], data_msft_final.index, frac=0.05)

# Plot the LOESS smoothing
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(data_msft_final.index, lowess[:, 1], label='LOESS Smoothing', color='red')
plt.title('MSFT Close Prices with LOESS Smoothing')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# EXPONENTIAL SMOOTHENING- A family of smoothing methods (including simple, double, and triple) that are widely used for short-term forecasting, particularly when dealing with trend and seasonality.

# %%
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Apply Triple Exponential Smoothing (Holt-Winters method)
triple_exp_smoothing = ExponentialSmoothing(data_msft_final['Close'],
                                            trend='add', # Try with "mul"
                                            seasonal='add',# Try with "mul"
                                            seasonal_periods=365).fit()

# Plot the original series and the smoothed series
plt.figure(figsize=(14, 7))
plt.plot(data_msft_final['Close'], label='MSFT Close Prices', color='blue')
plt.plot(triple_exp_smoothing.fittedvalues, label='Triple Exponential Smoothing', color='red')
plt.title('MSFT Close Prices with Triple Exponential Smoothing (Holt-Winters)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## TESTS FOR STATIONARITY
# 
# For a staionary data, its statistical properties (mean, variance, autocorrelation etc.) should remain constant over time.

# %%
from statsmodels.tsa.stattools import adfuller

# Perform the ADF test on the 'Close' prices of MSFT
adf_result = adfuller(data_msft_final['Close'])

# Output the test results
print("ADF Test Statistic: ", adf_result[0])
print("p-value: ", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"\t{key}: {value}")

# Interpretation based on p-value
if adf_result[1] < 0.05:
    print("Reject the null hypothesis (H0): The time series is stationary.")
else:
    print("Fail to reject the null hypothesis (H0): The time series is non-stationary.")


# %% [markdown]
# KPSS TEST-
# 
# KPSS test assumes the series is stationary under the null hypothesis and tests for stationarity around a deterministic trend.

# %%
from statsmodels.tsa.stattools import kpss

# Perform the KPSS test on the 'Close' prices of MSFT
kpss_result, kpss_pvalue, _, kpss_crit = kpss(data_msft_final['Close'], regression='c')

# Output the test results
print("KPSS Test Statistic: ", kpss_result)
print("p-value: ", kpss_pvalue)
print("Critical Values:")
for key, value in kpss_crit.items():
    print(f"\t{key}: {value}")

# Interpretation based on p-value
if kpss_pvalue < 0.05:
    print("Reject the null hypothesis (H0): The time series is non-stationary.")
else:
    print("Fail to reject the null hypothesis (H0): The time series is stationary.")


# %% [markdown]
# ADF and KPSS are often used together to confirm stationarity. If they give contradictory results, the series may be trend-stationary.

# %%
# Interpret both tests together
if adf_result[1] < 0.05 and kpss_pvalue >= 0.05:
    print("The series is likely stationary.")
elif adf_result[1] >= 0.05 and kpss_pvalue < 0.05:
    print("The series is likely non-stationary.")
elif adf_result[1] < 0.05 and kpss_pvalue < 0.05:
    print("The series may be trend-stationary (stationary after detrending).")
else:
    print("Results are inconclusive. Further investigation is needed.")


# %% [markdown]
# ## ACHIEVING STATIONARITY

# %%
# First differencing
data_msft_final['First_Diff'] = data_msft_final['Close'].diff()

# Second differencing (if necessary)
data_msft_final['Second_Diff'] = data_msft_final['First_Diff'].diff()

# Plotting the original series and the differenced series
plt.figure(figsize=(14, 7))

plt.subplot(3, 1, 1)
plt.plot(data_msft_final['Close'], label='Original Series')
plt.title('Original Series')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(data_msft_final['First_Diff'], label='First Differencing', color='orange')
plt.title('First Differencing')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(data_msft_final['Second_Diff'], label='Second Differencing', color='green')
plt.title('Second Differencing')
plt.grid(True)

plt.tight_layout()
plt.show()


# %%
# Perform ADF test on the first differenced series
adf_result_diff1 = adfuller(data_msft_final['First_Diff'].dropna())
print("ADF Test on First Differencing")
print("ADF Test Statistic: ", adf_result_diff1[0])
print("p-value: ", adf_result_diff1[1])
print("Critical Values:")
for key, value in adf_result_diff1[4].items():
    print(f"\t{key}: {value}")
if adf_result_diff1[1] < 0.05:
    print("The first differenced series is stationary based on the ADF test.\n")
else:
    print("The first differenced series is not stationary based on the ADF test.\n")

# Perform KPSS test on the first differenced series
kpss_result_diff1, kpss_pvalue_diff1, _, kpss_crit_diff1 = kpss(data_msft_final['First_Diff'].dropna(), regression='c')
print("KPSS Test on First Differencing")
print("KPSS Test Statistic: ", kpss_result_diff1)
print("p-value: ", kpss_pvalue_diff1)
print("Critical Values:")
for key, value in kpss_crit_diff1.items():
    print(f"\t{key}: {value}")
if kpss_pvalue_diff1 < 0.05:
    print("The first differenced series is non-stationary based on the KPSS test.\n")
else:
    print("The first differenced series is stationary based on the KPSS test.\n")


# %%
# Perform ADF test on the second differenced series
adf_result_diff2 = adfuller(data_msft_final['Second_Diff'].dropna())
print("ADF Test on Second Differencing")
print("ADF Test Statistic: ", adf_result_diff2[0])
print("p-value: ", adf_result_diff2[1])
print("Critical Values:")
for key, value in adf_result_diff2[4].items():
    print(f"\t{key}: {value}")
if adf_result_diff2[1] < 0.05:
    print("The second differenced series is stationary based on the ADF test.\n")
else:
    print("The second differenced series is not stationary based on the ADF test.\n")

# Perform KPSS test on the second differenced series
kpss_result_diff2, kpss_pvalue_diff2, _, kpss_crit_diff2 = kpss(data_msft_final['Second_Diff'].dropna(), regression='c')
print("KPSS Test on Second Differencing")
print("KPSS Test Statistic: ", kpss_result_diff2)
print("p-value: ", kpss_pvalue_diff2)
print("Critical Values:")
for key, value in kpss_crit_diff2.items():
    print(f"\t{key}: {value}")
if kpss_pvalue_diff2 < 0.05:
    print("The second differenced series is non-stationary based on the KPSS test.\n")
else:
    print("The second differenced series is stationary based on the KPSS test.\n")


# %%
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data_msft_final['Close'], model='additive', period=365)
trend = decomposition.trend
# Detrend by subtracting the trend component
data_msft_final['Detrended'] = data_msft_final['Close'] - trend
# Perform ADF test on the detrended series
adf_result_detrend = adfuller(data_msft_final['Detrended'].dropna())
print("ADF Test on Detrended Series")
print("ADF Test Statistic: ", adf_result_detrend[0])
print("p-value: ", adf_result_detrend[1])
print("Critical Values:")
for key, value in adf_result_detrend[4].items():
    print(f"\t{key}: {value}")
if adf_result_detrend[1] < 0.05:
    print("The detrended series is stationary based on the ADF test.\n")
else:
    print("The detrended series is not stationary based on the ADF test.\n")

# Perform KPSS test on the detrended series
kpss_result_detrend, kpss_pvalue_detrend, _, kpss_crit_detrend = kpss(data_msft_final['Detrended'].dropna(), regression='c')
print("KPSS Test on Detrended Series")
print("KPSS Test Statistic: ", kpss_result_detrend)
print("p-value: ", kpss_pvalue_detrend)
print("Critical Values:")
for key, value in kpss_crit_detrend.items():
    print(f"\t{key}: {value}")
if kpss_pvalue_detrend < 0.05:
    print("The detrended series is non-stationary based on the KPSS test.\n")
else:
    print("The detrended series is stationary based on the KPSS test.\n")


# %%
# Apply Logarithmic Transformation
data_msft_final['Log_Transformed'] = np.log(data_msft_final['Close'])
# Perform ADF test on the log-transformed series
adf_result_log = adfuller(data_msft_final['Log_Transformed'].dropna())
print("ADF Test on Log-Transformed Series")
print("ADF Test Statistic: ", adf_result_log[0])
print("p-value: ", adf_result_log[1])
print("Critical Values:")
for key, value in adf_result_log[4].items():
    print(f"\t{key}: {value}")
if adf_result_log[1] < 0.05:
    print("The log-transformed series is stationary based on the ADF test.\n")
else:
    print("The log-transformed series is not stationary based on the ADF test.\n")

# Perform KPSS test on the log-transformed series
kpss_result_log, kpss_pvalue_log, _, kpss_crit_log = kpss(data_msft_final['Log_Transformed'].dropna(), regression='c')
print("KPSS Test on Log-Transformed Series")
print("KPSS Test Statistic: ", kpss_result_log)
print("p-value: ", kpss_pvalue_log)
print("Critical Values:")
for key, value in kpss_crit_log.items():
    print(f"\t{key}: {value}")
if kpss_pvalue_log < 0.05:
    print("The log-transformed series is non-stationary based on the KPSS test.\n")
else:
    print("The log-transformed series is stationary based on the KPSS test.\n")


# %% [markdown]
# - First Differencing: Often enough to remove trends and achieve stationarity, as confirmed by both the ADF and KPSS tests.
# - Second Differencing: Applied if first differencing is insufficient; should achieve stationarity in most cases.
# - Detrending: Removes the trend component, making the series stationary if the trend is the primary cause of non-stationarity.
# - Logarithmic Transformation: Stabilizes the variance, useful for data exhibiting exponential growth or increasing variance over time. It generally does not make the data stationary.

# %% [markdown]
# ## PREDICTION OF HISTORICAL DATA USING RANDOM FOREST

# %%
# pip install ta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import ta
from datetime import date

symbols = ['MSFT', 'GOOG'] 
data = yf.download(symbols, start='2013-01-01', end=date.today())['Close']
data.columns = data.columns.get_level_values(0)
# Drop rows with missing values
data.dropna(inplace=True)

# Calculate daily returns
returns = data.pct_change().dropna()

# Create lag features
lags = [1, 2, 3, 5, 10]  # Lag by 1, 2, 3, 5, 10 days
lagged_features = pd.DataFrame(index=returns.index)

for lag in lags:
    lagged_features[f'MSFT_Lag_{lag}'] = returns['MSFT'].shift(lag)
    for col in ['GOOG']:
        lagged_features[f'{col}_Lag_{lag}'] = returns[col].shift(lag)

# Combine the lagged features with the returns dataframe
data_with_lags = pd.concat([returns, lagged_features], axis=1).dropna()

# Calculate rolling statistics (e.g., 10-day rolling mean and rolling std)
rolling_window = 10
data_with_lags['MSFT_Rolling_Mean'] = returns['MSFT'].rolling(window=rolling_window).mean()
data_with_lags['MSFT_Rolling_Std'] = returns['MSFT'].rolling(window=rolling_window).std()

# Add rolling statistics for other variables as well
for col in ['GOOG']:
    data_with_lags[f'{col}_Rolling_Mean'] = returns[col].rolling(window=rolling_window).mean()
    data_with_lags[f'{col}_Rolling_Std'] = returns[col].rolling(window=rolling_window).std()

# Calculate RSI for MSFT
data_with_lags['MSFT_RSI'] = ta.momentum.rsi(returns['MSFT'], window=14)

# Calculate MACD for MSFT
macd = ta.trend.MACD(returns['MSFT'])
data_with_lags['MSFT_MACD'] = macd.macd()
data_with_lags['MSFT_MACD_Signal'] = macd.macd_signal()

# Drop NaN values that result from technical indicators calculations
data_with_lags.dropna(inplace=True)

# Select the features and the target variable
features = data_with_lags.drop(columns=['MSFT'])
target = data_with_lags['MSFT']

# Standardize the features
scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42, shuffle=False)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual MSFT Prices', color='blue')
plt.plot(y_test.index, predictions, label='Predicted MSFT Prices', linestyle='--', color='red')
plt.legend()
plt.title('Actual vs Predicted MSFT Prices Using Random Forest')
plt.xlabel('Date')
plt.ylabel('MSFT Stock Price')
plt.show()


# %% [markdown]
# ## FORCASTING

# %% [markdown]
# ### ARIMA MODEL- AutoRegressive Integrated Moving Average
# ARIMA is best suited for forecasting non-seasonal, stationary data over shorter time periods.
# 
# AutoRegressive (AR): Uses past values to predict future values. p: Number of past values (lags) in AR.
# 
# Integrated (I): Applies differencing to make the data stationary (i.e., removing trends or seasonality). d: Number of differencing steps for stationarity.
# 
# Moving Average (MA): Uses past forecast errors to improve predictions. q: Number of past errors in MA.

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assume we've already differenced the series
differenced_series = data_msft_final['First_Diff'].dropna()

# Plot PACF to determine p
plt.figure(figsize=(10, 5))
plot_pacf(differenced_series, lags=20, method='ywm', ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF) - Determine p')
plt.show()

# Plot ACF to determine q
plt.figure(figsize=(10, 5))
plot_acf(differenced_series, lags=20, ax=plt.gca())
plt.title('Autocorrelation Function (ACF) - Determine q')
plt.show()

# %%
from statsmodels.tsa.arima.model import ARIMA

# Assuming p=0, d=1, q=0 based on the ACF/PACF analysis
p = 0
d = 1
q = 0

# Fit the ARIMA model
model = ARIMA(data_msft_final['Close'], order=(p, d, q))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())


# %%
# Plot residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 5))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.show()

# Plot ACF of residuals
plt.figure(figsize=(10, 5))
plot_acf(residuals, lags=20, ax=plt.gca())
plt.title('ACF of Residuals')
plt.show()

# Plot PACF of residuals
plt.figure(figsize=(10, 5))
plot_pacf(residuals, lags=20, method='ywm', ax=plt.gca())
plt.title('PACF of Residuals')
plt.show()


# %%
pip install pmdarima

# %%
import pmdarima as pm
from pmdarima import auto_arima

# %%
time_series = data_msft_final['Close']

# %%
# Using auto_arima to find the best fitting ARIMA model
auto_model = auto_arima(time_series,
                        start_p=0, start_q=0,
                        max_p=5, max_q=5,
                        m=1,  # Frequency of the series (m=1 for non-seasonal)
                        seasonal=False,  # Assuming non-seasonal data
                        d=None,  # Let auto_arima determine the value of d
                        trace=True,  # Print the output
                        error_action='ignore',  # Ignore non-fatal errors
                        suppress_warnings=True,  # Suppress warnings
                        stepwise=True)  # Use stepwise approach to search for best parameters

# Print the summary of the model
print(auto_model.summary())

# %%
# Fit the best ARIMA model identified by auto_arima
best_model = auto_model.fit(time_series)

# Print the summary of the best model 
print(best_model.summary())


# %%
# Forecast the next 10 periods
n_periods = 10
forecast, conf_int = best_model.predict(n_periods=n_periods, return_conf_int=True)

# Print the forecasted values and confidence intervals
print("Forecasted values:", forecast)
print("Confidence intervals:")
print(conf_int)

# Plot the forecasted values along with the confidence intervals
plt.figure(figsize=(10, 5))
plt.plot(time_series.index, time_series, label='Historical Data')
plt.plot(pd.date_range(time_series.index[-1], periods=n_periods, freq='B'), forecast, label='Forecast')
plt.fill_between(pd.date_range(time_series.index[-1], periods=n_periods, freq='B'),
                 conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()



