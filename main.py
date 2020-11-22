#%%
import numpy as np
import kmeans
import common
import naive_em
import em

#%%
X = np.loadtxt("toy_data.txt")
for K in range(1,5):
    for seed in range(0,5):

        title = "K=" + str(K) + ", seed=" + str(seed)

        M, P = common.init(X, K, seed)
        cost = kmeans.run(X, M, P)
        print(title, cost[2])

#common.plot(X, M, P, title)

# %%
