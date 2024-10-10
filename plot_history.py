import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
from utils.parser import plot_parser
import os



parser = plot_parser()
args = parser.parse_args()

exp = args.exp
exps = os.listdir("Experiments")
if exp not in exps :
    raise ValueError(f"There is no Experiement {exp} to test")

history = pd.read_csv(os.path.join("Experiments",exp,"history.csv"))
metrices = list(history.columns)[2:] # remove epoch from columns

group = dict()
for metric in metrices:
    metric_type = metric.split("@")[0]
    if metric_type not in group:
        group[metric_type]=[]
    group[metric_type].append(metric)


x = np.arange(history.shape[0])

fig , axes = plt.subplots(2,2,figsize=(15,10))
axes = axes.flatten()
axes[0].set_title(f"Loss over epochs")
axes[0].set_xlabel(f"epochs")
axes[0].set_ylabel(f"Bpr loss")
axes[0].plot(x,history["Train Loss"].values)

for i,(metric_name,metric_k) in enumerate(group.items()):
    axes[i+1].set_title(f"{metric_name} over epochs")
    axes[i+1].set_xlabel("epochs")
    axes[i+1].set_ylabel(f"{metric_name}@k")
    for k in metric_k:
        axes[i+1].plot(x,history[k],label=k)
    axes[i+1].legend()
# plt.tight_layout() 
plt.show()



















