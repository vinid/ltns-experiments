import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({'font.size': 22})

loaded_data = pd.read_csv("performance_results/performance_1", names = ["pred", "const", "time"])

loaded_data = loaded_data.groupby(['pred', 'const'],as_index=False).mean()

data = (np.array(loaded_data["time"].values).reshape((5,5)))


fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.set(font_scale=2)

sns.heatmap(data, annot=True, xticklabels=["4", "8", "12", "20", "30"],yticklabels=["4", "8", "12", "20", "30"], cmap="Blues")
ax.set_xlabel("Number of constants", fontsize=20)
ax.set_ylabel("Number of predicates (arity 1)", fontsize=20)
plt.yticks(rotation=0)
fig.savefig('performance_1.pdf',bbox_inches='tight')
plt.show()



