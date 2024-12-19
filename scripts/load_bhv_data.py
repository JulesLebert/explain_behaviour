
from sys import platform
from pathlib import Path
from pynwb import NWBHDF5IO, TimeSeries
import yaml
import pandas as pd
from tqdm import tqdm

def get_trial_table(nwb_file):
    """
    This function extracts the trial table from a NWB file.
    :param nwb_file:
    :return:
    """
    io = NWBHDF5IO(path=nwb_file, mode='r')
    nwb_data = io.read()
    nwb_objects = nwb_data.objects
    objects_list = [data for key, data in nwb_objects.items()]
    data_to_take = None

    # Iterate over NWB objects but keep "trial"
    for obj in objects_list:
        if 'trial' in obj.name:
            data = obj
            if isinstance(data, TimeSeries):
                continue
            else:
                data_to_take = data
                break
        else:
            continue
    trial_data_frame = data_to_take.to_dataframe()

    return trial_data_frame

def windows_to_mac_path(
        path_name, 
        base_dir = '/Volumes',
        ):
    parts = path_name.split('\\')
    mac_path = Path(base_dir, *parts[3:])
    return mac_path

def main():
    group = Path('/Users/lebert/home/code/context/explain_behaviour/group.yaml')
    with open(group, 'r') as f:
        sessions = yaml.safe_load(f)
    expert_sessions = sessions['NWB_CI_LSENS']['Context_expert_sessions']
    # sess = expert_sessions[0][1]
    df = pd.DataFrame()
    for _, session_path in tqdm(expert_sessions):
        if platform == 'darwin': # On MacOS
            session_path = windows_to_mac_path(session_path)

        df_sess = get_trial_table(session_path)
        df = pd.concat([df, df_sess])
    print(df)


    print(sessions)

if __name__ == '__main__':
    main()