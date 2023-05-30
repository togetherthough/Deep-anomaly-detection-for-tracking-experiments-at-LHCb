import json
import glob
import csv
import pandas as pd

src = "SimFiles_WithChi2/"

files = glob.glob('SimFiles_WithChi2/*', recursive=True)

data = []

for single_file in files:
    with open(single_file, 'r') as f:

        # Use 'try-except' to skip files that may be missing data
        try:
            json_file = json.load(f)
            for i in range(len(json_file)):

                observation = json_file[i]
                print(observation.keys())
                data.append([
                    observation['is_ghost'],
                    observation['chi2'],

                    observation['nb_hits'],
                    observation['qop'],
                    observation['tx'],
                    observation['ty'],
                    observation['x'],
                    observation['y'],
                    observation['z'],
                ])
        except KeyError:
            print(f'Skipping {single_file}')


data.insert(0, ['isghost', "chi2", 'nb_hits',
            'qop', 'tx', 'ty', 'x', 'y', 'z'])

# Export to CSV.
# Add the date to the file name to avoid overwriting it each time.
csv_filename = 'SimData.csv'
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

df = pd.read_csv('SimData.csv')
print(df.head())
