# Credit Risk Modeling - Logistic Regression

Keyword:: PD Model, ROC-AUC: 0.86
Skills used:: Logistic Regression

This project implements a Probability of Default (PD) model using Logistic Regression on the Lending Club dataset. Under the IRB (Internal Ratings-Based) approach, banks are permitted to develop their own models to estimate borrower PD using internal historical data, enabling more efficient capital allocation and a competitive advantage over the standardized baseline. 

Logistic regression remains the industry standard for credit risk modeling. While tree-based methods (Random Forest, XGBoost) can achieve higher classification performance, logistic regression is preferred for its interpretability, which is a regulatory requirement under the Basel framework.

### 1. Preprocessing Data

***→ Handle Large Number of Features:*** The Lending Club Dataset contains 2260668 rows and 145 columns, which is a very large amount of features. To process this, I divided it into 3 data types:

- Categorical (dtype = object)
- Numerical (dtype = numeric or float)
- Datetime (dtype = object but had the suffix _d or _date in column name)

and processed each data type separately.

***General cleaning:** 

***→ Near Zero-variance Problem:** A feature with a dominant value accounting for the majority of the data. Some columns have mostly missing value or a strongly dominant value (>99% of data) like total_rec_late_fee 99% are zeros and will not give much predictive power. I decided to also drop column where >90% data are null because it’s also a type of near-zero variance (with a dominant value which is null, so even if we impute, it’s still the same problem).*

#### 1.1 Preprocess Categorical Data

***→ High Cardinality Problem:** A feature with high cardinality mean it has too many unique values and unlikely to give predictive signals. It also gives too many variables when we later use on-hot-encoding. So I filter out variables like emp_title, desc (loan description) , title (purpose description). These variables contain free-text, cardinality of >500k (having 500k+ unique values) and requires NLP techniques to process.*

For other categorical columns, I used Weight of Evidence (WoE) (with adjustment to avoid divide by zero) and Information Value (IV) with Coarse Classing to judge which feature contribute meaningfully to separating the labels. 

I used the benchmark 0.02 < IV < 1, which mean features with normal - strong predictive signal in industry standard to select the features.

![image.png](Credit%20Risk%20Modeling%20-%20Logistic%20Regression/image.png)

Those that I kept are: ['term', 'home_ownership', 'verification_status', 'initial_list_status', 'application_type']

![Screenshot 2026-04-17 at 21.29.14.png](Credit%20Risk%20Modeling%20-%20Logistic%20Regression/Screenshot_2026-04-17_at_21.29.14.png)

****sub_grade is not included in the model to avoid information leakage***

#### 1.2 Preprocess Date Data

Set a reference data and turn them to month_since_ to convert to numeric data type.

#### 1.3 Preprocess Numeric Data

***→ Multicollinearity Problem:*** Feature pairs (groups) with correlation > 0.85 were identified; the most representative feature from each group (highest sum of pairwise correlations) was retained.

WoE/IV was applied using 10-bin splits (benchmark: 0.03 < IV < 0.5), yielding 28 numeric features. Optimal binning was then determined for each variable using a Decision Tree Classifier to identify split points that maximize separation.

![Screenshot 2026-04-17 at 21.48.59.png](Credit%20Risk%20Modeling%20-%20Logistic%20Regression/Screenshot_2026-04-17_at_21.48.59.png)

![Screenshot 2026-04-17 at 21.49.45.png](Credit%20Risk%20Modeling%20-%20Logistic%20Regression/Screenshot_2026-04-17_at_21.49.45.png)

### 2. Modeling

For target variable, a loan is labelled as default if is has been late for over 90 days or been charged off.

Logistic Regression was trained with L2 regularization (C=10, solver=lbfgs), identified via RandomizedSearchCV. Class imbalance was addressed using `class_weight='balanced'` since resampling techniques (SMOTE, under/oversampling) yielded inferior performance.

**Results:** ROC-AUC: 0.862 | KS Statistic: 0.562

At a decision threshold of 0.3: Precision (default class): 0.24 | Recall (default class): 0.90

```
              precision    recall  f1-score   support

           0       0.98      0.59      0.74    394814
           1       0.24      0.90      0.38     56611

    accuracy                           0.63    451425
   macro avg       0.61      0.75      0.56    451425
weighted avg       0.88      0.63      0.69    451425
```

![Screenshot 2026-04-17 at 21.55.16.png](Credit%20Risk%20Modeling%20-%20Logistic%20Regression/Screenshot_2026-04-17_at_21.55.16.png)

# Variable Dictionary

| # | Variable Name | Data Type | Description | Category |
| --- | --- | --- | --- | --- |
| 1 | id | string | Unique ID assigned for the specific loan application | Identifiers |
| 2 | member_id | string | Unique "Customer Number" (to see if one person has multiple loans) | Identifiers |
| 3 | loan_amnt | numeric | The amount the borrower **asked for** | Loan Info |
| 4 | funded_amnt | numeric | The amount the bank **approved** and promised to fund | Loan Info |
| 5 | funded_amnt_inv | numeric | The portion of the loan actually **funded by investors** (the P2P part) | Loan Info |
| 6 | term | categorical | Term of the loan (36 or 60 months) | Loan Info |
| 7 | int_rate | numeric | Interest rate on the loan | Loan Info |
| 8 | installment | numeric | The **Fixed Monthly Payment** the borrower must pay | Loan Info |
| 9 | grade | categorical | Lending Club assigned loan grade (A–G) | Loan Info |
| 10 | sub_grade | categorical | Lending Club assigned loan subgrade (A1–G5) | Loan Info |
| 11 | emp_title | string | Job title supplied by the borrower | Borrower Profile |
| 12 | emp_length | categorical | Employment length in years (0 = <1 year, 10 = 10+ years) | Borrower Profile |
| 13 | home_ownership | categorical | Home ownership status: RENT, OWN, MORTGAGE, OTHER | Borrower Profile |
| 14 | annual_inc | numeric | Self-reported annual income provided by the borrower | Borrower Profile |
| 15 | verification_status | categorical | Whether income was verified by Lending Club, not verified, or source verified (Lending Club **verified that the source of the income exists**, but they did not necessarily confirm the exact amount of the income) | Borrower Profile |
| 16 | issue_d | date | Date the loan was funded | Loan Info |
| 17 | loan_status | categorical | Current status of the loan (target variable) | Loan Info |
| 18 | pymnt_plan | categorical | Whether a payment plan has been put in place for the loan | Loan Info |
| 19 | url | string | URL for the Lending Club page with listing data | Identifiers |
| 20 | desc | string | Loan description provided by the borrower (free text) | Borrower Profile |
| 21 | purpose | categorical | Borrower-provided category for the loan request | Loan Info |
| 22 | title | string | Loan title provided by the borrower (free text) | Loan Info |
| 23 | zip_code | string | First 3 digits of the borrower's zip code | Borrower Profile |
| 24 | addr_state | categorical | State provided by the borrower in the loan application | Borrower Profile |
| 25 | dti | numeric | Debt-to-income ratio (excluding mortgage and Lending Club loan) | Borrower Financials |
| 26 | delinq_2yrs | numeric | Number of 30+ day delinquency incidents in the borrower's credit file for the past 2 years | Credit History |
| 27 | earliest_cr_line | date | Date the borrower's earliest reported credit line was opened | Credit History |
| 28 | fico_range_low | numeric | Lower boundary of the borrower's **FICO** range at loan origination | Credit History |
| 29 | fico_range_high | numeric | Upper boundary of the borrower's **FICO** range at loan origination | Credit History |
| 30 | inq_last_6mths | numeric | Number of inquiries in the past 6 months (excluding auto and mortgage) | Credit History |
| 31 | mths_since_last_delinq | numeric | Months since the borrower's last delinquency | Credit History |
| 32 | mths_since_last_record | numeric | Months since the last public record | Credit History |
| 33 | open_acc | numeric | Number of open credit lines in the borrower's credit file (only active account) | Credit History |
| 34 | pub_rec | numeric | Number of derogatory public records ***(A public record in a credit context typically refers to legal or government-related filings that suggest financial distress)*** | Credit History |
| 35 | revol_bal | numeric | Total credit revolving balance | Credit History |
| 36 | revol_util | numeric | Revolving line utilization rate (balance / total available revolving credit) | Credit History |
| 37 | total_acc | numeric | Total number of credit lines currently in the borrower's credit file (Every account ever opened) | Credit History |
| 38 | initial_list_status | categorical | Initial listing status of the loan (W = whole, F = fractional) | Loan Info |
| 39 | out_prncp | numeric | Remaining outstanding principal for total amount funded | Payment History |
| 40 | out_prncp_inv | numeric | Remaining outstanding principal for investor-funded portion | Payment History |
| 41 | total_pymnt | numeric | Payments received to date for total amount funded | Payment History |
| 42 | total_pymnt_inv | numeric | Payments received to date for investor-funded portion | Payment History |
| 43 | total_rec_prncp | numeric | Principal received to date | Payment History |
| 44 | total_rec_int | numeric | Interest received to date | Payment History |
| 45 | total_rec_late_fee | numeric | Late fees received to date | Payment History |
| 46 | recoveries | numeric | Post charge-off gross recovery amount | Payment History |
| 47 | collection_recovery_fee | numeric | Post charge-off collection fee | Payment History |
| 48 | last_pymnt_d | date | Date when last payment was received | Payment History |
| 49 | last_pymnt_amnt | numeric | Last total payment amount received | Payment History |
| 50 | next_pymnt_d | date | Next scheduled payment date | Payment History |
| 51 | last_credit_pull_d | date | The most recent date LendingClub **pulled the borrower's credit report** for this specific loan | Credit History |
| 52 | last_fico_range_low | numeric | Lower boundary of the borrower's last FICO pulled | Credit History |
| 53 | last_fico_range_high | numeric | Upper boundary of the borrower's last FICO pulled
***(* context: Lenders don't just check credit score once. They often pull updates to monitor for signs of distress (like a sudden drop in FICO) while the loan is active)*** | Credit History |
| 54 | collections_12_mths_ex_med | numeric | Number of collections in 12 months excluding medical collections | Credit History |
| 55 | mths_since_last_major_derog | numeric | **Months since the most recent "Major Derogatory" event** (specifically a 90-day or worse rating). | Credit History |
| 56 | policy_code | categorical | Publicly available policy_code=1; new products not publicly available policy_code=2 | Loan Info |
| 57 | application_type | categorical | Whether loan is individual or joint application with co-borrowers | Borrower Profile |
| 58 | annual_inc_joint | numeric | Combined self-reported annual income for co-borrowers | Borrower Profile |
| 59 | dti_joint | numeric | DTI ratio for co-borrowers (excluding mortgages and LC loan) | Borrower Financials |
| 60 | verification_status_joint | categorical | Whether the co-borrowers' joint income was verified by LC | Borrower Profile |
| 61 | acc_now_delinq | numeric | Number of accounts on which the borrower is currently delinquent | Credit History |
| 62 | tot_coll_amt | numeric | Total collection amounts ever owed | Credit History |
| 63 | tot_cur_bal | numeric | Total current balance of all accounts | Borrower Financials |
| 64 | open_acc_6m | numeric | Number of trade lines (total loans) in the last 6 months | Credit History |
| 65 | open_act_il | numeric | Number of currently active installment trade lines | Credit History |
| 66 | open_il_12m | numeric | Number of installment accounts opened in the past 12 months | Credit History |
| 67 | open_il_24m | numeric | Number of installment accounts opened in the past 24 months | Credit History |
| 68 | mths_since_rcnt_il | numeric | Months since most recent installment account was opened | Credit History |
| 69 | total_bal_il | numeric | Total current balance of all installment accounts | Borrower Financials |
| 70 | il_util | numeric | Ratio of total current balance to high credit/limit on all installment accounts | Borrower Financials |
| 71 | open_rv_12m | numeric | Number of revolving trade lines opened in the past 12 months | Credit History |
| 72 | open_rv_24m | numeric | Number of revolving trade lines opened in the past 24 months | Credit History |
| 73 | max_bal_bc | numeric | highest balance a borrower has ever reached on any of their individual bank-issued credit card | Borrower Financials |
| 74 | all_util | numeric | Ratio of balance to credit limit on all trade lines | Borrower Financials |
| 75 | total_rev_hi_lim | numeric | Total revolving high credit/credit limit | Borrower Financials |
| 76 | inq_fi | numeric | Number of personal finance inquiries | Credit History |
| 77 | total_cu_tl | numeric | Number of accounts (trade lines) the borrower has specifically with **Credit Unions** | Credit History |
| 78 | inq_last_12m | numeric | Number of credit inquiries in the past 12 months | Credit History |
| 79 | acc_open_past_24mths | numeric | Number of trade lines opened in the past 24 months | Credit History |
| 80 | avg_cur_bal | numeric | Average current balance of all accounts | Borrower Financials |
| 81 | bc_open_to_buy | numeric | Total open-to-buy (available credit) on revolving bankcards | Borrower Financials |
| 82 | bc_util | numeric | Ratio of total current balance to credit limit for all bankcard accounts | Borrower Financials |
| 83 | chargeoff_within_12_mths | numeric | Number of charge-offs within 12 months | Credit History |
| 84 | delinq_amnt | numeric | Past-due amount owed for accounts on which the borrower is currently delinquent | Credit History |
| 85 | mo_sin_old_il_acct | numeric | Months since oldest bank installment account was opened | Credit History |
| 86 | mo_sin_old_rev_tl_op | numeric | Months since oldest revolving account was opened | Credit History |
| 87 | mo_sin_rcnt_rev_tl_op | numeric | Months since most recent revolving account was opened | Credit History |
| 88 | mo_sin_rcnt_tl | numeric | Months since most recent account was opened | Credit History |
| 89 | mort_acc | numeric | Number of mortgage accounts | Credit History |
| 90 | mths_since_recent_bc | numeric | Months since most recent bankcard account was opened | Credit History |
| 91 | mths_since_recent_bc_dlq | numeric | Months since most recent bankcard delinquency | Credit History |
| 92 | mths_since_recent_inq | numeric | Months since most recent inquiry | Credit History |
| 93 | mths_since_recent_revol_delinq | numeric | Months since most recent revolving delinquency | Credit History |
| 94 | num_accts_ever_120_pd | numeric | Number of accounts ever 120 or more days past due | Credit History |
| 95 | num_actv_bc_tl | numeric | Number of currently active bankcard accounts | Credit History |
| 96 | num_actv_rev_tl | numeric | Number of currently active revolving trade lines | Credit History |
| 97 | num_bc_sats | numeric | Number of satisfactory bankcard accounts (***a satisfactory account is one that is not delinquent and is active)*** | Credit History |
| 98 | num_bc_tl | numeric | Number of bankcard accounts | Credit History |
| 99 | num_il_tl | numeric | Number of installment accounts | Credit History |
| 100 | num_op_rev_tl | numeric | Number of open revolving accounts | Credit History |
| 101 | num_rev_accts | numeric | Number of revolving accounts | Credit History |
| 102 | num_rev_tl_bal_gt_0 | numeric | Number of revolving trade lines with balance > 0 | Credit History |
| 103 | num_sats | numeric | Number of satisfactory accounts | Credit History |
| 104 | num_tl_120dpd_2m | numeric | Number of accounts currently 120+ days past due (updated in past 2 months) | Credit History |
| 105 | num_tl_30dpd | numeric | Number of accounts currently 30 days past due (updated in past 2 months) | Credit History |
| 106 | num_tl_90g_dpd_24m | numeric | Number of accounts 90+ days past due in the last 24 months | Credit History |
| 107 | num_tl_op_past_12m | numeric | Number of accounts opened in the past 12 months | Credit History |
| 108 | pct_tl_nvr_dlq | numeric | Percent of trades never delinquent | Credit History |
| 109 | percent_bc_gt_75 | numeric | Percentage of all bankcard accounts > 75% utilization | Borrower Financials |
| 110 | pub_rec_bankruptcies | numeric | Number of public record bankruptcies | Credit History |
| 111 | tax_liens | numeric | Number of tax liens | Credit History |
| 112 | tot_hi_cred_lim | numeric | Total high credit/credit limit across all accounts | Borrower Financials |
| 113 | total_bal_ex_mort | numeric | Total credit balance excluding mortgage | Borrower Financials |
| 114 | total_bc_limit | numeric | Total bankcard high credit/credit limit | Borrower Financials |
| 115 | total_il_high_credit_limit | numeric | Total installment high credit/credit limit | Borrower Financials |
| 116 | revol_bal_joint | numeric | Sum of revolving credit balance of co-borrowers, net of duplicate balances | Borrower Financials |
| 117 | sec_app_fico_range_low | numeric | FICO range (low) for the secondary applicant | Credit History |
| 118 | sec_app_fico_range_high | numeric | FICO range (high) for the secondary applicant | Credit History |
| 119 | sec_app_earliest_cr_line | date | Earliest credit line at time of application for the secondary applicant | Credit History |
| 120 | sec_app_inq_last_6mths | numeric | Credit inquiries in the last 6 months for the secondary applicant | Credit History |
| 121 | sec_app_mort_acc | numeric | Number of mortgage accounts for the secondary applicant | Credit History |
| 122 | sec_app_open_acc | numeric | Number of open trades for the secondary applicant | Credit History |
| 123 | sec_app_revol_util | numeric | Revolving utilization ratio for the secondary applicant | Borrower Financials |
| 124 | sec_app_open_act_il | numeric | Number of currently active installment trades for the secondary applicant | Credit History |
| 125 | sec_app_num_rev_accts | numeric | Number of revolving accounts for the secondary applicant | Credit History |
| 126 | sec_app_chargeoff_within_12_mths | numeric | Number of charge-offs within last 12 months for the secondary applicant | Credit History |
| 127 | sec_app_collections_12_mths_ex_med | numeric | Number of collections in last 12 months (ex. medical) for the secondary applicant | Credit History |
| 128 | sec_app_mths_since_last_major_derog | numeric | Months since most recent 90-day or worse rating for the secondary applicant | Credit History |
| 129 | hardship_flag | categorical | Whether the borrower is on a hardship plan | Hardship / Settlement |
| 130 | hardship_type | categorical | Type of hardship plan offered | Hardship / Settlement |
| 131 | hardship_reason | categorical | Reason the hardship plan was offered | Hardship / Settlement |
| 132 | hardship_status | categorical | Status of the hardship plan (active, pending, canceled, completed, broken) | Hardship / Settlement |
| 133 | deferral_term | numeric | Number of months the borrower is expected to pay less due to hardship plan | Hardship / Settlement |
| 134 | hardship_amount | numeric | Interest payment the borrower committed to making monthly while on hardship plan | Hardship / Settlement |
| 135 | hardship_start_date | date | Start date of the hardship plan period | Hardship / Settlement |
| 136 | hardship_end_date | date | End date of the hardship plan period | Hardship / Settlement |
| 137 | payment_plan_start_date | date | Date the first hardship plan payment is due | Hardship / Settlement |
| 138 | hardship_length | numeric | Number of months the borrower will make smaller payments due to hardship plan | Hardship / Settlement |
| 139 | hardship_dpd | numeric | Account days past due as of the hardship plan start date | Hardship / Settlement |
| 140 | hardship_loan_status | categorical | Loan status as of the hardship plan start date | Hardship / Settlement |
| 141 | orig_projected_additional_accrued_interest | numeric | Original projected additional interest to accrue for the hardship payment plan | Hardship / Settlement |
| 142 | hardship_payoff_balance_amount | numeric | Payoff balance amount as of the hardship plan start date | Hardship / Settlement |
| 143 | hardship_last_payment_amount | numeric | Last payment amount as of the hardship plan start date | Hardship / Settlement |
| 144 | disbursement_method | categorical | Method by which the borrower receives their loan (CASH or DIRECT_PAY) | Loan Info |
| 145 | debt_settlement_flag | categorical | Whether the charged-off borrower is working with a debt settlement company | Hardship / Settlement |

---