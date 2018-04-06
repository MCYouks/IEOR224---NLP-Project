## Visualizing the stock market structure
## Source: http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#stock-market

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold


def retry(f, n_attempts=3):
    "Wrapper function to retry function calls in case of exceptions"
    def wrapper(*args, **kwargs):
        for i in range(n_attempts):
            try:
                return f(*args, **kwargs)
            except Exception:
                if i == n_attempts - 1:
                    raise
    return wrapper

def timer(f):
    "Measures the execution time of the method/function"
    def wrapper(*args, **kwargs):
        start_time = time.time()
        output = f(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        print('Execution time : %.1f seconds' % elapsed_time)
        
        return output
    
    return wrapper

@retry
def dataReader(symbol, data_source, start, end):
    """Get the historical data from Yahoo finance.
    
    retry decorator is used because quotes_historical_google can temporarily fail
    for various reasons (e.g. empty result from Google API).

    Parameters
    ----------
    symbol : str
        Ticker symbol to query for, for example ``"DELL"``.
    start : datetime.datetime
        Start date.
    end : datetime.datetime
        End date.

    Returns
    -------
    X : dataframe
        The columns are ``date`` -- date, ``open``, ``high``,
        ``low``, ``close`` and ``volume`` of type float.
    """
    return web.DataReader(symbol, data_source, start, end)


start_date = datetime(2003, 1, 1).date()
end_date = datetime(2008, 1, 1).date()

@timer
def quotes_historical_yahoo(symbols, start=start_date, end=end_date, 
                            progress=True):
    """Get the historical data from Yahoo finance and separate the
    successful requests from the rejected ones. """
    accepted_symbol = []
    rejected_symbol = []
    quotes = []
    
    for i, symbol in enumerate(symbols):
        if progress:
            progress_rate = int(i / len(symbols) * 100)
            print('# Progress :', str(progress_rate) + '%')
        try:
            df = dataReader(symbol, 'yahoo', start, end)
            accepted_symbol.append(symbol)
            quotes.append(df)
        except Exception:
            rejected_symbol.append(symbol)
            
    return quotes, accepted_symbol, rejected_symbol
    

symbol_dict = {
    'TOT': 'Total',
    'XOM': 'Exxon',
    'CVX': 'Chevron',
    'COP': 'ConocoPhillips',
    'VLO': 'Valero Energy',
    'MSFT': 'Microsoft',
    'IBM': 'IBM',
    'TWX': 'Time Warner',
    'CMCSA': 'Comcast',
    'CVC': 'Cablevision',
    'YHOO': 'Yahoo',
    'DELL': 'Dell',
    'HPQ': 'HP',
    'AMZN': 'Amazon',
    'TM': 'Toyota',
    'CAJ': 'Canon',
    'SNE': 'Sony',
    'F': 'Ford',
    'HMC': 'Honda',
    'NAV': 'Navistar',
    'NOC': 'Northrop Grumman',
    'BA': 'Boeing',
    'KO': 'Coca Cola',
    'MMM': '3M',
    'MCD': 'McDonald\'s',
    'PEP': 'Pepsi',
    'K': 'Kellogg',
    'UN': 'Unilever',
    'MAR': 'Marriott',
    'PG': 'Procter Gamble',
    'CL': 'Colgate-Palmolive',
    'GE': 'General Electrics',
    'WFC': 'Wells Fargo',
    'JPM': 'JPMorgan Chase',
    'AIG': 'AIG',
    'AXP': 'American express',
    'BAC': 'Bank of America',
    'GS': 'Goldman Sachs',
    'AAPL': 'Apple',
    'SAP': 'SAP',
    'CSCO': 'Cisco',
    'TXN': 'Texas Instruments',
    'XRX': 'Xerox',
    'WMT': 'Wal-Mart',
    'HD': 'Home Depot',
    'GSK': 'GlaxoSmithKline',
    'PFE': 'Pfizer',
    'SNY': 'Sanofi-Aventis',
    'NVS': 'Novartis',
    'KMB': 'Kimberly-Clark',
    'R': 'Ryder',
    'GD': 'General Dynamics',
    'RTN': 'Raytheon',
    'CVS': 'CVS',
    'CAT': 'Caterpillar',
    'DD': 'DuPont de Nemours'}

quotes, accepted_symbol, rejected_symbol = quotes_historical_yahoo(symbol_dict.keys())

success_rate = int(len(accepted_symbol) / len(symbol_dict.keys()) * 100)        
print('Success rate :', str(success_rate) + '%')

variation = [df['Close'] - df['Open'] for df in quotes]
variation = pd.DataFrame(variation).T
variation.columns = accepted_symbol
print(variation.head())

# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy()
X /= X.std(axis=0)
edge_model.fit(X)

# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()
names = np.array([symbol_dict[symbol] for symbol in accepted_symbol])

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))


# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.show()


