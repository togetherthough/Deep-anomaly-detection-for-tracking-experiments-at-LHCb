import json
import glob
import csv
import pandas as pd

src = "data/"

files = glob.glob(src + '*', recursive=True)


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
                    observation['trk_best_pt'],
                    observation['trk_best_qop'],
                    observation['trk_chi2'],
                    observation['trk_chi2T'],
                    observation['trk_chi2V'],
                    observation['trk_first_qop'],
                    observation['trk_kalman_docaz'],
                    observation['trk_kalman_ip'],
                    observation['trk_kalman_ip_chi2'],
                    observation['trk_kalman_ipx'],
                    observation['trk_kalman_ipy'],
                    observation['trk_ndof'],
                    observation['trk_ndofT'],
                    observation['trk_ndofV'],
                    observation['trk_velo_docaz'],
                    observation['trk_velo_ip'],
                    observation['trk_velo_ip_chi2'],
                    observation['trk_velo_ipx'],
                    observation['trk_velo_ipy'],
                ])

        except KeyError:
            print(f'Skipping {single_file}')


data.insert(0, ['isghost', 'chi2', 'nb_hits', 'qop', 'tx', 'ty', 'x', 'y', 'z', 'trk_best_pt', 'trk_best_qop', 'trk_chi2', 'trk_chi2T',
                 'trk_chi2V', 'trk_first_qop', 'trk_kalman_docaz', 'trk_kalman_ip', 'trk_kalman_ip_chi2', 'trk_kalman_ipx', 'trk_kalman_ipy',
                 'trk_ndof', 'trk_ndofT', 'trk_ndofV', 'trk_velo_docaz', 'trk_velo_ip', 'trk_velo_ip_chi2', 'trk_velo_ipx', 'trk_velo_ipy'])

# Export to CSV.
# Add the date to the file name to avoid overwriting it each time.
csv_filename = 'SimulationData.csv'
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

df = pd.read_csv('SimulationData.csv')
print(df.head())
