import numpy as np
from scipy import stats

# Data Generator
N = 10
a = np.random.randn(N) + 2
b = np.random.randn(N)

# Calculate t Statistics
var_a = a.var(ddof=1)
var_b = b.var(ddof=1)
s = np.sqrt((var_a + var_b) / 2)
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N))
df = 2*N - 2
p = 1 - stats.t.cdf(t, df=df)

print("t: ", t, " p: ", 2*p) # two-sided

t2, p2 = stats.ttest_ind(a, b)
print("t: ", t, " p: ", p2)