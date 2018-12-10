import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("./train.csv")
df.info()

# %% Delete irrelevant attributes
df = df.drop(columns=["PassengerId", "Ticket", "Name"])

# %% Overall survivors statistics
total = df.shape[0]
survivors = df["Survived"].sum()
non_survivors = total - df["Survived"].sum()

print("Surivors: %d, Non-survivors: %d" % (survivors, non_survivors))

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x="Survived")
plt.title("Survival count")

plt.subplot(1, 2, 2)
plt.pie([survivors, non_survivors], labels=["Survived", "Not survived"], autopct="%1.0f")
plt.title("Survival ratio")

plt.show()

# %% Different classes analysis
df_class = df[["Pclass", "Survived"]].groupby(["Pclass"]).count()
upper_people = df_class.iloc[0]
middle_people = df_class.iloc[1]
lower_people = df_class.loc[2]

print("Upper class: %d, Middle class: %d, Lower class: %d" % (upper_people, middle_people, lower_people))

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x="Pclass")
plt.title("Pclass count")

plt.subplot(1, 2, 2)
plt.pie(df_class, labels=['1', '2', '3'], autopct="%1.0f")
plt.title("Pclass ratio")

plt.show()

df_survivors = (df[df["Survived"] == 1])[["Pclass", "Survived"]].groupby(["Pclass"]).count()
upper_survivors = df_survivors.iloc[0]
middle_survivors = df_survivors.iloc[1]
lower_survivors = df_survivors.iloc[2]

print("Upper class survivors: %d, Middle class survivors: %d, Lower class survivors: %d" % (upper_survivors, middle_survivors, lower_survivors))

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
sns.countplot(data=df[df["Survived"] == 1], x="Pclass")
plt.title("Survivors count")

plt.subplot(1, 2, 2)
plt.pie(df_survivors, labels=['1', '2', '3'], autopct="%1.0f")
plt.title("Survivors ratio")

plt.show()

# %% Different ages analysis
average_age = df["Age"].mean()
std_age = df["Age"].std()
missing_age = df["Age"].isnull().sum()

random = np.random.randint(average_age - std_age, average_age + std_age, size=missing_age)
df["Age"][np.isnan(df["Age"])] = random

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
df["Age"].hist(bins=70)
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
df.boxplot(column="Age", showfliers=False)
df["Age"].describe()

# %% Correlation between age and survival
df2 = df.groupby(["Sex", "Survived"])["Survived"].count()
