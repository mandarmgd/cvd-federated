import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
heartbeat_dir = os.path.join(script_dir, '..', 'datasets', 'Heartbeats')

set_a_path = os.path.join(heartbeat_dir, 'set_a.csv')
set_b_path = os.path.join(heartbeat_dir, 'set_b.csv')

df_a = pd.read_csv(set_a_path)
df_b = pd.read_csv(set_b_path)

df_combined = pd.concat([df_a, df_b], ignore_index=True)
df_combined = df_combined[~df_combined['sublabel'].str.contains('noisy', case = False, na = False)]
df_combined = df_combined.drop(columns = ['dataset', 'sublabel'])

mapping = {
    'normal': 0,
    'murmur': 1,
    'extrahls': 1,
    'artifact': 1,
    'extrastole': 2
}
df_combined['label'].str.lower()
df_combined['sublabel'] = df_combined['label'].map(mapping).astype('Int64')

output_path = os.path.join(heartbeat_dir, 'combined.csv')
df_combined.to_csv(output_path, index=False)

# non_null_count = df['label'].notnull().sum()
# null_count = df['label'].isnull().sum()
# print(non_null_count, null_count)



