## Visualizing the stock market structure
## Source: http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#stock-market

import os
import pandas as pd
import numpy as np
from sklearn import cluster, covariance, manifold

def pandasReader(io):
    """Return a series of the weekly variation of historical data."""
    # Open the file
    df = pd.read_csv(io)
    
    # Set time column as index and make sure it is a date format
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    
    # Extract the Close column only
    df = df[['Open', 'Close']]
    
    # Opening price on Monday
    df1 = df.groupby(pd.Grouper(freq='W-MON'))[['Open']].first()
    
    # We sync the df1 index on Friday to easily merge it with the closing date dataframe
    df1 = df1.groupby(pd.Grouper(freq='W-FRI'))[['Open']].first()
    
    # Closing price on Friday
    df2 = df.groupby(pd.Grouper(freq='W-FRI'))[['Close']].last()
    
    # Concatenate those 2 frames
    df3 = pd.concat([df1, df2], axis=1)
    
    # Create a variation column
    df3['Variation'] = df3['Close'] - df3['Open']
    
    # Create a return column
    df3['%Return'] = df3['Variation'].pct_change()
    
    # Create a rolling median column
    df3['Median'] = df3['%Return'].rolling(2, min_periods=1).median()
    
    # Create a return column that replaces inf values by the rolling median
    df3['Return'] = np.where(df3['%Return'].isin([-np.inf, np.inf]), df3['Median'], df3['%Return'])
        
    return df3[['Return']] 


def quotesReader(path, progress=False, saveExcel=False):
    """Get the historical weekly returns from data in various csv files. """
    symbols = []
    frames = []
    
    # List of csv files we are going to exclude from our analysis
    rejected_symbols = ['NLSN', 'QRVO', 'CHTR', 'HCA', 'CBOE', 'CFG', 'PSX', 'IQV', 'NAVI',
                        'NWS', 'FTV', 'MPC', 'KHC', 'ALLE', 'ABBV', 'KMI', 'APTV', 'KORS', 'GM',
                        'VRSK', 'PYPL', 'AVY', 'WRK', 'ZTS', 'DG', 'HPE', 'TRIP', 'XYL', 'FBHS',
                        'CBRE', 'HII', 'HST', 'EVHC', 'COTY', 'AIV', 'AVGO', 'UA', 'LYB', 'SYF',
                        'INFO', 'NCLH', 'AIG', 'NWSA', 'HLT']
    
    # Open each csv file in path
    for i, filename in enumerate(os.listdir(path)):
        
        # Print progress rate
        if progress:
            progress_rate = int(i / len(os.listdir(path)) * 100)
            print('# Progress :', str(progress_rate) + '%')
            
        
        # Extract the symbol from the filename
        symbol = filename.split('.csv')[0]
        
        # Only take into account the non-rejected symbols
        if not symbol in rejected_symbols:
            
            # Add it to the symbol list
            symbols.append(symbol)
            
            # Get the path of the csv file
            filepath = os.path.join(path, filename)

            # Extract the weekly returns using our pandasReader function and add it to the list of frames
            df = pandasReader(filepath)
            frames.append(df)
        
        
    # Concat all the frames horizontally and name the columns with the corresponding symbols
    df1 = pd.concat(frames, axis=1)
    df1.columns = symbols
    
    # Remove rows containing NaN values
    df1 = df1.dropna(axis=0)
    
    # Save as excel file
    if saveExcel:
        writer = pd.ExcelWriter('output.xlsx')
        df1.to_excel(writer,'Sheet1')
        writer.save()
    
    return df1


df = quotesReader('Stocks')

# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = df.copy()
X /= X.std(axis=0)
edge_model.fit(X)


# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()
names = np.array(df.columns)

Clusters = dict()
for i in range(n_labels + 1):
    Clusters[i+1] = names[labels == i].tolist()

print(Clusters)