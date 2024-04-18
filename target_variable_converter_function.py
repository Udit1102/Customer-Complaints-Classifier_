from preprocessing import preprocess_text
import pandas as pd

def categorizer(i):
	if i == "Credit reporting, credit repair services, or other personal consumer reports":
		return int(0)
	elif i == "Debt collection":
		return int(1)
	elif i == "Consumer Loan":
		return int(2)
	elif i == "Mortgage":
		return int(3)
