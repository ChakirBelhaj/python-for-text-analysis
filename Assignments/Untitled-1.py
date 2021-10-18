df = pd.read_csv (r'./KS_model_training_data.csv', sep = ',')
df = df.drop(columns=['backers_count', 'converted_pledged_amount', 'pledged', 'usd_pledged'])
df = df.dropna()

df['created_at'] = pd.to_datetime(df['created_at'],unit='s')
df['created_hour'] = df.created_at.apply(lambda x: x.hour)
df['created_month'] = df.created_at.apply(lambda x: x.month)
df['created_year'] = df.created_at.apply(lambda x: x.year)


df['deadline'] = pd.to_datetime(df['deadline'],unit='s')
df['deadline_hour'] = df.deadline.apply(lambda x: x.hour)
df['deadline_month'] = df.deadline.apply(lambda x: x.month)
df['deadline_year'] = df.deadline.apply(lambda x: x.year)


df['launched_at'] = pd.to_datetime(df['launched_at'],unit='s')
df['launched_hour'] = df.launched_at.apply(lambda x: x.hour)
df['launched_month'] = df.launched_at.apply(lambda x: x.month)
df['launched_year'] = df.launched_at.apply(lambda x: x.year)


df['launched_after_deadline'] = (df['launched_at'] > df['deadline'])
# 
# df['deadline_at'] = pd.to_datetime(df['deadline'],unit='s')
# df['deadline_hour'] = df.deadline.apply(lambda x: x.hour)


df["category"] = df["category"].astype('category')
df["category_encoded"] = df["category"].cat.codes



df["subcategory"] = df["subcategory"].astype('category')
df["subcategory_encoded"] = df["subcategory"].cat.codes


df["location"] = df["location"].astype('category')
df["location_encoded"] = df["location"].cat.codes


df["country"] = df["country"].astype('category')
df["country_encoded"] = df["country"].cat.codes

df['blurb_length'] = df['blurb'].str.len()
df['name_length'] = df['name'].str.len()



print(np.corrcoef(df['funded'], df['category_encoded']))
print(np.corrcoef(df['funded'], df['subcategory_encoded']))
print(np.corrcoef(df['funded'], df['location_encoded']))
print(np.corrcoef(df['funded'], df['country_encoded']))
print(np.corrcoef(df['funded'], df['blurb_length']))
print(np.corrcoef(df['funded'], df['fx_rate']))
print(np.corrcoef(df['funded'], df['name_length']))

# print(df.head(10))

# print(np.corrcoef(df['funded'], df['staff_pick']))
# print(np.corrcoef(df['funded'], df['created_month']))
# print(np.corrcoef(df['funded'], df['deadline']))
# df['category'].value_counts()[:20].plot(kind='bar')
# df.plot.scatter(x = 'goal', y ='category')




