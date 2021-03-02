log = "run.sh.log"
with open(log,'r') as f:
  train_epochs = []
  cv_epochs = []
  for line in f:
    line = line.replace("\n","");
    if line.startswith("Train Summary"):
      cells = line.split(" | ")
      train_epochs.append([int(cells[1].replace("End of Epoch ","")),float(cells[2][:-2].replace("Time ","")),
      float(cells[3].replace("Train Loss ",""))])
    elif line.startswith("Valid Summary"):
      cells = line.split(" | ")
      cv_epochs.append([int(cells[1].replace("End of Epoch ","")),float(cells[2][:-2].replace("Time ","")),
      float(cells[3].replace("Valid Loss ",""))])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ["Epoch","Time","Loss"]
train_df = pd.DataFrame(data=np.array(train_epochs),columns = cols)
cv_df = pd.DataFrame(data=np.array(cv_epochs),columns = cols)

plt.figure(0)
plt.plot(train_df["Epoch"],train_df["Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss")
plt.savefig("train_loss.png")

plt.figure(1)
plt.plot(cv_df["Epoch"],cv_df["Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.savefig("cv_loss.png")

