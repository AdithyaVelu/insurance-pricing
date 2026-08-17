import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import joblib

df_freq = pd.read_csv("freq_data.csv")
df_sev = pd.read_csv("severity_data.csv")

df_freq['Exposure'] = df_freq['Exposure'].clip(upper=1.0)
df_freq['ClaimNb'] = df_freq['ClaimNb'].clip(upper=4)

actuarial_group = df_freq.groupby('Area')[['ClaimNb', 'Exposure']].sum().reset_index()
actuarial_group['true_freq'] = actuarial_group['ClaimNb'] / actuarial_group['Exposure']

#plt.figure(figsize=(10, 5)) 
#sns.barplot(x='Area', y='true_freq', data = actuarial_group)
#plt.show()

df_sev = df_sev.groupby('IDpol')['ClaimAmount'].sum().reset_index()

df = pd.merge(df_freq, df_sev, on='IDpol', how='left')
df['ClaimAmount'] = df['ClaimAmount'].fillna(0)
df_crash = df[df['ClaimAmount']>0]

#plt.figure(figsize=(10, 5)) 
#plt.xlim(0, 20000)
#sns.histplot(x='ClaimAmount', data=df_crash, bins=50, binrange=(0, 20000))
#plt.show()

categ_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
df_encoded = pd.get_dummies(df, columns=categ_cols, drop_first=True)

df_encoded['Is_Young'] = (df_encoded['DrivAge'] < 25).astype(int)
df_encoded['Density_Log'] = np.log(df_encoded['Density'] + 1)
df_encoded = df_encoded.drop(columns=['Density'])
age_bins = [17, 24, 34, 44, 54, 64, 74, 120]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
df_encoded['Age_Band'] = pd.cut(df_encoded['DrivAge'], bins=age_bins, labels=age_labels)


'''
plt.figure(figsize=(20, 5)) 
sns.barplot(x='DrivAge', y='ClaimNb', data=df_encoded)
plt.show()
'''

X = df_encoded.drop(columns=['ClaimNb', 'ClaimAmount', 'IDpol'])
y = df_encoded['ClaimNb']
exposure = df_encoded['Exposure']
X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, y, exposure, test_size=0.20, random_state=42
)
if 'Age_Band' in X_train.columns:
    X_train = pd.get_dummies(X_train, columns=['Age_Band'], drop_first=True)

X_train = X_train.astype(float)
X_train_sm = sm.add_constant(X_train)

poisson_model = sm.GLM(
    endog=y_train, 
    exog=X_train_sm, 
    offset=np.log(exp_train), 
    family=sm.families.Poisson()
)

poisson_results = poisson_model.fit()

coefficients = poisson_results.params
multipliers = np.exp(coefficients)
pricing_table = pd.DataFrame({
    'Raw_Coefficient': coefficients,
    'Pricing_Multiplier': multipliers
})

pricing_table = pricing_table.sort_values(by='Pricing_Multiplier', ascending=False)
#print(pricing_table)

if 'Age_Band' in X_test.columns:
    X_test = pd.get_dummies(X_test, columns=['Age_Band'], drop_first=True)

X_test = X_test.astype(float)
X_test_sm = sm.add_constant(X_test)
X_test_sm = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0)

predicted_claims = poisson_results.predict(exog=X_test_sm, offset=np.log(exp_test))

actual_total = y_test.sum()
predicted_total = predicted_claims.sum()
#print('actual = ',actual_total, ', predicted = ', predicted_total)

error_margin = abs(actual_total - predicted_total) / actual_total * 100
#print('Model Error Margin:', error_margin)


df_crash = df_encoded[df_encoded['ClaimAmount'] > 0].copy()
df_crash['Avg_Severity'] = df_crash['ClaimAmount'] / df_crash['ClaimNb']

X_sev = df_crash.drop(columns=['ClaimNb', 'ClaimAmount', 'IDpol', 'Avg_Severity', 'Exposure'])
y_sev = df_crash['Avg_Severity']

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
    X_sev, y_sev, test_size=0.20, random_state=42
)
if 'Age_Band' in X_train_sev.columns:
    X_train_sev = pd.get_dummies(X_train_sev, columns=['Age_Band'], drop_first=True)

X_train_sev = X_train_sev.astype(float)
X_train_sev_sm = sm.add_constant(X_train_sev)

gamma_model = sm.GLM(
    endog=y_train_sev, 
    exog=X_train_sev_sm, 
    family=sm.families.Gamma(link=sm.families.links.Log())
)

gamma_results = gamma_model.fit()
#print(gamma_results.summary())

annual_freq = poisson_results.predict(exog=X_test_sm)

X_test_sev_sm = X_test_sm.reindex(columns=gamma_results.model.exog_names, fill_value=0)
expected_sev = gamma_results.predict(exog=X_test_sev_sm)

X_test['Expected_Freq'] = annual_freq
X_test['Expected_Sev'] = expected_sev
X_test['Pure_Premium'] = annual_freq * expected_sev

#print(X_test[['Expected_Freq', 'Expected_Sev', 'Pure_Premium']].head())

fixed_expenses = 50.00         # £50 flat fee to keep the lights on
variable_expense_pct = 0.15    # 15% for commissions and taxes
profit_margin_pct = 0.05       # 5% target profit

X_test['Gross_Premium'] = (X_test['Pure_Premium'] + fixed_expenses) / (1 - (variable_expense_pct + profit_margin_pct))
X_test['Gross_Premium'] = X_test['Gross_Premium'].round(2)
print(X_test[['Pure_Premium', 'Gross_Premium']].head())

MIN_PREMIUM = 150.00   
MAX_PREMIUM = 4500.00

X_test['Final_Street_Price'] = X_test['Gross_Premium'].clip(lower=MIN_PREMIUM, upper=MAX_PREMIUM)

capped_high = len(X_test[X_test['Gross_Premium'] > MAX_PREMIUM])
capped_low = len(X_test[X_test['Gross_Premium'] < MIN_PREMIUM])
print(f"Drivers artificially bumped up to Minimum (${MIN_PREMIUM}): {capped_low}")
print(f"Drivers artificially reduced to Maximum (${MAX_PREMIUM}): {capped_high}")

#print(X_test[['Gross_Premium', 'Final_Street_Price']].head(10))

joblib.dump(poisson_results, 'poisson_freq_model.pkl')
joblib.dump(gamma_results, 'gamma_sev_model.pkl')
X_test.to_csv('final_pricing_portfolio.csv', index=True)


