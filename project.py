import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# ============start functions=================================================================
def change_columns_type_and_fillna(df):
    penalty_all_columns = list(df.columns[31:61])
    other_bool_colums = [df.columns[i] for i in [11, 20, 24, 66, 67, 69, 70, 71, 72, 74]]
    bool_columns=set_col_list(other_bool_colums[0:3],penalty_all_columns,other_bool_colums[3:])
    float_columns = 'tr_amount_applied0 tr_amount_applied1 tr_amount_applied2 tr_amount_applied3 tr_area tr_number_parcels tr_payment_actual0 tr_payment_actual1 tr_payment_actual2 tr_payment_actual3 tr_penalty_amount0 tr_penalty_amount1 tr_penalty_amount2 tr_penalty_amount3 tr_risk_factor'.split()

    for c in bool_columns:
        df[c]=df[c].map({'true': 1, 'false': 0})

    for col in float_columns:
        df[col] = df[col].replace("", 0)

    df[float_columns] = df[float_columns].astype(float)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    zipped_actual_applied = zip(float_columns[0:4], float_columns[6:10])

    for index, values in enumerate(zipped_actual_applied):
        df['diff_{}'.format(index)] = df[values[0]] - df[values[1]]

    return df


# --> function for create oone list (get as many list as we want and merged them into one list - usually used for select rellevant collumns)
def set_col_list(*args):
    col_list = []
    for arg in args:
        col_list.extend(arg)
    return col_list

def create_summarized_columns(**kwargs):
    for k, v in kwargs.items():
        if len(v) == 2:
            df[k] = df.loc[:, v[0]:v[1]].sum(axis=1)
        else:
            df[k] = df.loc[:, v].sum(axis=1)


def create_confidence_interval(df, col1, val, col2):
    l = df[df[col1] == val][col2]
    sample_size = l.count()
    sample_mean = l.mean()

    z_critical = stats.norm.ppf(q=0.975)  # Get the z-critical value*

    pop_stdev = l.std()  # Get the population standard deviation

    margin_of_error = z_critical * (pop_stdev / math.sqrt(sample_size))

    confidence_interval = (sample_mean - margin_of_error,
                           sample_mean + margin_of_error)
    return confidence_interval


def drop_exreamly_values(df, column_name):
    total_applications = df[column_name].count()
    critic_delta = 0.05 * total_applications
    delta = total_applications

    # while condition aimed for prevent us drop too much data cut also drop as much as we can outliers numeric values which afected on the standard deviation
    while delta >= critic_delta:
        total_events_before = df[column_name].count()
        mean = df[column_name].mean()
        std = df[column_name].std()
        upper_bound = mean + (2 * std)
        lower_bound = mean - (2 * std)
        df = df[(df[column_name] > lower_bound) & (df[column_name] < upper_bound)]
        total_events_after = df[column_name].count()
        delta = total_events_before - total_events_after
    return df


def q1(df, columns, target):

    y = df[target]
    x = df[columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    lm = LinearRegression()
    lm.fit(x_train, y_train)

    print('intercept : ', lm.intercept_)

    cdf = pd.DataFrame(lm.coef_, index=x.columns, columns=['coeff'])
    print(cdf)

    prediction = lm.predict(x_test)

    print('MAE : ', metrics.mean_absolute_error(y_test, prediction))
    print('MSE : ', metrics.mean_squared_error(y_test, prediction))
    print('RMRE : ', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
    print('Explained variance : ', metrics.explained_variance_score(y_test, prediction))

    plt.scatter(y_test, prediction)
    plt.xlabel('Y Test (True Values)')
    plt.ylabel('Predicted Values')
    plt.show()


def q2(df):
    years = ['2015', '2016', '2017']
    conf_interval_list = []
    for year in years:
        conf_interval_list.append(create_confidence_interval(df_by_year, 'tr_year', year, 'Total_panelty_numbers'))
    return conf_interval_list


def q3(df, columns, target):

    y = df[target]
    x = df[columns]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    log_model = LogisticRegression()
    log_model.fit(x_train, y_train)
    predictions = log_model.predict(x_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))



# ============end functions=================================================================


# ===============code====================================================================


df = pd.read_pickle('Payment_application_df.pickle')
df=change_columns_type_and_fillna(df)

sever_panelty_columns = ['tr_penalty_B3', 'tr_penalty_B4', 'tr_penalty_B5', 'tr_penalty_B6', 'tr_penalty_B16',
                   'tr_penalty_BGK', 'tr_penalty_C16', 'tr_penalty_JLP3', 'tr_penalty_V5', 'tr_penalty_BGP',
                   'tr_penalty_BGKV', 'tr_penalty_B5F']
create_summarized_columns(Total_amount_apllied=['tr_amount_applied0','tr_amount_applied3'],Total_payment_actual=['tr_payment_actual0','tr_payment_actual3'],Total_panelty_amount=['tr_penalty_amount0','tr_penalty_amount3'],Total_panelty_numbers=['tr_penalty_ABP','tr_penalty_V5'],Total_sever_panelty_numbers=sever_panelty_columns)
df['diff_Toal_aplied_actual']=df['Total_amount_apllied']-df['Total_payment_actual']
df['diff_0_is_negative_or_0']=np.where(df['diff_0']<=0, 1, 0)

id_col = ['tr_applicant', 'tr_application']
amounts = ['tr_penalty_amount0', 'tr_penalty_amount1', 'tr_penalty_amount2', 'tr_penalty_amount3']

df_gb=df.groupby(by=['tr_applicant', 'tr_application'])
df_gb_first_lines=df_gb.first()
df_time_delta=df.groupby(by=['tr_applicant', 'tr_application'])['time:timestamp'].apply(lambda x: (x.max()-x.min()).days).reset_index(name='Time_delta')
df_gb_first_lines=pd.merge(df_gb_first_lines.reset_index(),df_time_delta,on=['tr_applicant', 'tr_application'])


# Q1 Answer:
mask_for_q1=df_gb_first_lines['diff_0']>0
df_mask_for_q1=df_gb_first_lines[mask_for_q1]
df_for_q1=drop_exreamly_values(df_mask_for_q1,'diff_0')
print('Q1 Answer: \n')
q1(df_for_q1,['diff_0'],'Total_panelty_amount')
print('\n------------------------------------\n')

# Q2 Answer:

df_by_year=df.groupby(by=['tr_year','tr_application']).first().reset_index()
result2=q2(df_by_year)
print('Q2 Answer: \n')
print(result2)
print('\n------------------------------------\n')


#  Q3 Answer:
print('Q3 Answer: \n')
q3(df_gb_first_lines,['diff_0','tr_number_parcels','Time_delta','Total_panelty_amount'],'tr_rejected')
print('\n------------------------------------\n')