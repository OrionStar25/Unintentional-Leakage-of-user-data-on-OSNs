
# coding: utf-8

import pandas as pd

cf_twitter_data = pd.read_csv("data/twitter_gender.csv", encoding='latin1')

gender_m_df = cf_twitter_data[
    (cf_twitter_data['gender:confidence'] == 1) & 
    (
        (cf_twitter_data['gender'] == 'male')
    )
]

gender_f_df = cf_twitter_data[
    (cf_twitter_data['gender:confidence'] == 1) & 
    (
        (cf_twitter_data['gender'] == 'female')
    )
]

username_m_df = gender_m_df[['name', 'gender']]
username_f_df = gender_f_df[['name', 'gender']]

username_f_df = username_f_df[-100:]
username_m_df = username_m_df[-100:]
print(len(username_f_df))
print(len(username_m_df))

username_both_df = username_f_df.append(username_m_df)
print(len(username_both_df))

username_both_df.to_csv('test_data.csv', encoding='utf-8')