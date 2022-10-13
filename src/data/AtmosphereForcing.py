import xarray as xr
from utils import check_input, get_configs
from data import simulations, times
import os
cfg = get_configs()

class AtmosphereForcing:
    def __init__(self, path):
        self.path = path
        # self.path = cfg['data']['path']
        # self.simulation = simulation
        # self.time_frame = time_frame

        # check_input(simulation, simulations)
        # check_input(time_frame, times)
        # simulation_run = simulation.split('_')[0]
        # print(os.path.join(self.path, f'Atompshere_Forcing/{simulation}/Regridded_8km/{simulation_run}_8km_anomaly_{simulation_run}'))
        self.data = xr.open_dataset(path, decode_times=False)
        # self.data = xr.open_dataset()

    def aggregate_dims(self,):
        dims = self.data.dims
        if 'time' in dims:
            self.data = self.data.mean(dim='time')
        if 'nv4' in dims:
            self.data = self.data.mean(dim='nv4')
        return self

    def save_as_df(self):
        csv_path = f"{self.path[:-3]}.csv"
        df = self.data.to_dataframe()
        df.to_csv(csv_path)
        return self


# forcing = AtmosphereForcing(simulation='ccsm4_rcp2.6', time_frame='1995-2100')
# forcing = AtmosphereForcing(path=r"/users/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing/ccsm4_rcp2.6/Regridded_8km/CCSM4_8km_anomaly_rcp26_1995-2100.nc")
# print(forcing.data)
# stop = []