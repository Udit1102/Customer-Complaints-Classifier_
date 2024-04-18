import pandas as pd
from target_variable_converter_function import categorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned_complaints_dataset.csv")
print(df["product"].value_counts(normalize=True))

x = df.iloc[:,1] + " " + df.iloc[:,2] + " " + df.iloc[:,3]+ " " + df.iloc[:,4]
y = df["product"].apply(categorizer)
#print(x.count(), y.count())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

'''product
Credit reporting, credit repair services, or other personal consumer reports    0.693083
Debt collection                                                                 0.175516
Mortgage                                                                        0.121317
Consumer Loan                                                                   0.010084
N'''