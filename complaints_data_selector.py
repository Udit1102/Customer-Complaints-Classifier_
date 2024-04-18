import pandas as pd
import numpy as np

df = pd.read_csv("complaints.csv")
important_product = ["Consumer Loan", "Mortgage", "Debt collection", "Credit reporting, credit repair services, or other personal consumer reports"]

#df_prop = df[df["Product"].isin(important_product)].iloc[:, 1:6]
#print(df_prop["Product"].value_counts(normalize=True))

df_row = df.sample(frac=0.002, random_state=7)
df_modified = df_row[df_row["Product"].isin(important_product)].iloc[:, 1:6]
df_modified.columns = df_modified.columns.str.lower()
df_modified.fillna("Unknown", inplace=True)
df_modified.to_csv("cleaned_complaints_dataset.csv", index=False)
print(df_modified.count())

#result for comparision of information present after and before reduction of dataset
'''Credit reporting, credit repair services, or other personal consumer reports    0.687012
Debt collection                                                                 0.177094
Mortgage                                                                        0.125869
Consumer Loan                                                                   0.010024
N'''