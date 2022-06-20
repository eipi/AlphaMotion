import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({'X': [1.1, 2.05, 3.07, 4.2],
                   'Y1': [10.1, 15.2, 35.3, 40.4],
                   'Y2': [55.05, 40.4, 84.17, 31.5]})
print(df)

df.set_index('X', inplace=True)
print(df)

Xresampled = np.linspace(1, 4, 15)
print(Xresampled)

# Resampling
# df = df.reindex(df.index.union(resampling))

# Interpolation technique to use. One of:

# 'linear': Ignore the index and treat the values as equally spaced. This is the only method supported on MultiIndexes.
# 'time': Works on daily and higher resolution data to interpolate given length of interval.
# 'index', 'values': use the actual numerical values of the index.
# 'pad': Fill in NaNs using existing values.
# 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial': Passed to scipy.interpolate.interp1d. These methods use the numerical values of the index. Both 'polynomial' and 'spline' require that you also specify an order (int), e.g. df.interpolate(method='polynomial', order=5).
# 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima': Wrappers around the SciPy interpolation methods of similar names. See Notes.
# 'from_derivatives': Refers to scipy.interpolate.BPoly.from_derivatives which replaces 'piecewise_polynomial' interpolation method in scipy 0.18.

df_resampled = df.reindex(df.index.union(Xresampled)).interpolate('values').loc[Xresampled]
print(df_resampled)

# gca stands for 'get current axis'
ax = plt.gca()
df.plot(style='X', y='Y2', color='blue', ax=ax, label='Original Data')
df_resampled.plot(style='.-', y='Y2', color='red', ax=ax, label='Interpolated Data')
ax.set_ylabel('Y1')
plt.show()