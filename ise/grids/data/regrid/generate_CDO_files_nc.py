import numpy as np
import netCDF4 as nc
import os

def polarstereo_inv(x, y, a=6378137.0, e=0.08181919, phi_c=-70, lambda_0=0):
    # Convert to radians
    phi_c = np.radians(phi_c)
    lambda_0 = np.radians(lambda_0)

    # If the standard parallel is in S.Hemi., switch signs
    if phi_c < 0:
        pm = -1
        phi_c = -phi_c
        lambda_0 = -lambda_0
        x = -x
        y = -y
    else:
        pm = 1

    # Calculate intermediate variables
    t_c = np.tan(np.pi/4 - phi_c/2) / ((1 - e * np.sin(phi_c)) / (1 + e * np.sin(phi_c)))**(e/2)
    m_c = np.cos(phi_c) / np.sqrt(1 - e**2 * np.sin(phi_c)**2)
    rho = np.sqrt(x**2 + y**2)
    t = rho * t_c / (a * m_c)

    # Calculate phi using a series
    chi = np.pi/2 - 2 * np.arctan(t)
    phi = chi + (e**2/2 + 5*e**4/24 + e**6/12 + 13*e**8/360) * np.sin(2*chi) \
        + (7*e**4/48 + 29*e**6/240 + 811*e**8/11520) * np.sin(4*chi) \
        + (7*e**6/120 + 81*e**8/1120) * np.sin(6*chi) \
        + (4279*e**8/161280) * np.sin(8*chi)

    lambda_val = lambda_0 + np.arctan2(x, -y)

    # Correct the signs and phasing
    phi = pm * phi
    lambda_val = pm * lambda_val
    lambda_val = (lambda_val + np.pi) % (2 * np.pi) - np.pi  # Want longitude in the range -pi to pi

    # Convert back to degrees
    phi = np.degrees(phi)
    lambda_val = np.degrees(lambda_val)

    return phi, lambda_val



def wnc(var, fname, vname, uname, lname, dnames, add_singleton_time_dim, ncformat):
    """
    Write data from workspace to netCDF file.
    Syntax:
    wnc(var, fname, vname, uname, lname, dnames, add_singleton_time_dim, ncformat)
    var = variable array
    fname = name of netcdf file (in quotations, i.e. 'example.nc')
    vname = name of variable (also in quotations)
    uname = name of variable units (also in quotations)
    lname = long variable name
    dnames = names of dimensions
    add_singleton_time_dim = 0/1 to not add/add a singleton time dimension
    ncformat = netcdf file format
    """

    if isinstance(var, list) or isinstance(var, tuple):
        var = np.array(var)

    if var.ndim == 1:
        dnames = [dnames]
        nDims = 1
        var_dims = len(var)
    else:
        nDims = var.ndim
        var_dims = var.shape
        
    if len(dnames) == 3 and list(dnames)[0] != 'nv4':
        dnames = ['nv4', 'y', 'x']

    if len(dnames) != nDims:
        raise ValueError('Dimension name list not equal in size to dimensionality of variable')

    DimInput = {}
    for n in range(nDims):
        DimInput[str(list(dnames)[n])] = var_dims[n]

    if add_singleton_time_dim:
        # if no time dimension, add an unlimited time dimension
        if 'time' not in dnames:
            DimInput['time'] = None
        else:
            raise ValueError('You requested to add unlimited time dim, but a time dim already exists.')
    
    dataset = nc.Dataset(fname, mode='w', format=ncformat)
    
    dataset.createDimension('x', int(DimInput['x']))
    dataset.createDimension('y', int(DimInput['y']))
    if 'nv4' in DimInput.keys():
        dataset.createDimension('nv4', int(DimInput['nv4']))
        if DimInput['nv4'] != 4:
            raise ValueError('nv4 must be 4, something is wrong.')
    
    # dataset.createDimension('y', int(DimInput['y']))

    
    dataset.createVariable(vname, var.dtype, DimInput)
    dataset[vname][:] = var
    dataset[vname].units = uname
    dataset[vname].long_name = lname
    dataset.close()


def generate_CDO_files_nc(grid, proj_info, output_data_type, output_text=False):
    # Make X,Y cartesian coordinates
    dx = grid['dx']
    dy = grid['dy']
    nx_centers = grid['nx_centers']
    ny_centers = grid['ny_centers']
    nsize = grid['nx_centers'] * grid['ny_centers']

    # Create lat,lon coordinates
    xcenters, ycenters = np.meshgrid(np.arange(ny_centers) * dy, np.arange(nx_centers) * dx)
    LI_grid_center_lat, LI_grid_center_lon = polarstereo_inv(
        xcenters.flatten() - proj_info['falseeasting'],
        ycenters.flatten() - proj_info['falsenorthing'],
        proj_info['earthradius'], proj_info['eccentricity'],
        proj_info['standard_parallel'],
        proj_info['longitude_rot']
    )

    LI_grid_center_lat = LI_grid_center_lat.reshape(ycenters.shape)
    LI_grid_center_lon = LI_grid_center_lon.reshape(xcenters.shape)

    xcorners, ycorners = np.meshgrid(np.arange(ny_centers + 1) * dy - dy / 2, np.arange(nx_centers + 1) * dx - dx / 2)
    LI_grid_corner_lat, LI_grid_corner_lon = polarstereo_inv(
        xcorners.flatten() - proj_info['falseeasting'],
        ycorners.flatten() - proj_info['falsenorthing'],
        proj_info['earthradius'], proj_info['eccentricity'],
        proj_info['standard_parallel'],
        proj_info['longitude_rot']
    )

    LI_grid_corner_lat = LI_grid_corner_lat.reshape(ycorners.shape)
    LI_grid_corner_lon = LI_grid_corner_lon.reshape(xcorners.shape)

    # Generate 3d corner coordinates
    LI_grid_center_lat_CDO_format = LI_grid_center_lat
    LI_grid_center_lon_CDO_format = LI_grid_center_lon

    LI_grid_corner_lat_CDO_format = np.zeros(LI_grid_center_lat_CDO_format.shape + (4,))
    NEcorner_lat = LI_grid_corner_lat[1:, :-1]
    LI_grid_corner_lat_CDO_format[:, :, 0] = NEcorner_lat
    SEcorner_lat = LI_grid_corner_lat[1:, 1:]
    LI_grid_corner_lat_CDO_format[:, :, 1] = SEcorner_lat
    SWcorner_lat = LI_grid_corner_lat[:-1, 1:]
    LI_grid_corner_lat_CDO_format[:, :, 2] = SWcorner_lat
    NWcorner_lat = LI_grid_corner_lat[:-1, :-1]
    LI_grid_corner_lat_CDO_format[:, :, 3] = NWcorner_lat

    LI_grid_corner_lon_CDO_format = np.zeros(LI_grid_center_lon_CDO_format.shape + (4,))
    NEcorner_lon = LI_grid_corner_lon[1:, :-1]
    LI_grid_corner_lon_CDO_format[:, :, 0] = NEcorner_lon
    SEcorner_lon = LI_grid_corner_lon[1:, 1:]
    LI_grid_corner_lon_CDO_format[:, :, 1] = SEcorner_lon
    SWcorner_lon = LI_grid_corner_lon[:-1, 1:]
    LI_grid_corner_lon_CDO_format[:, :, 2] = SWcorner_lon
    NWcorner_lon = LI_grid_corner_lon[:-1, :-1]
    LI_grid_corner_lon_CDO_format[:, :, 3] = NWcorner_lon

    if output_data_type == 'radians':
        LI_grid_center_lat_CDO_format = np.deg2rad(LI_grid_center_lat_CDO_format)
        LI_grid_center_lon_CDO_format = np.deg2rad(LI_grid_center_lon_CDO_format)
        LI_grid_corner_lat_CDO_format = np.deg2rad(LI_grid_corner_lat_CDO_format)
        LI_grid_corner_lon_CDO_format = np.deg2rad(LI_grid_corner_lon_CDO_format)

    LI_grid_dims_CDO_format = np.array(LI_grid_center_lat_CDO_format.shape, dtype=np.int32)
    LI_grid_imask_CDO_format = np.zeros(LI_grid_dims_CDO_format, dtype=np.int32)
    LI_grid_imask_CDO_format[:, :] = 1

    # Write CDO grid netcdf file
    
    print(f"Generating {grid['LatLonOutputFileName']}")

    if os.path.exists(grid['LatLonOutputFileName']):
        os.remove(grid['LatLonOutputFileName'])

        
    # wnc(LI_grid_center_lat,grid['LatLonOutputFileName'],'lat','degrees_north','grid center latitude',{'x','y'},0,'NETCDF4')    
    # wnc(LI_grid_center_lon,grid['LatLonOutputFileName'],'lon','degrees_east','grid center longitude',{'x','y'},0,'NETCDF4')
    # # bounds
    # wnc(LI_grid_corner_lat_CDO_format,grid['LatLonOutputFileName'],'lat_bnds','degrees_north','grid corner latitude',{'nv4','x','y'},0,'NETCDF4')
    # wnc(LI_grid_corner_lon_CDO_format,grid['LatLonOutputFileName'],'lon_bnds','degrees_east','grid corner longitude',{'nv4','x','y'},0,'NETCDF4')

    # # dummy needed for mapping
    # wnc((LI_grid_center_lon*0+1).astype(int),grid['LatLonOutputFileName'],'dummy','1','dummy variable',{'x','y'},0,'NETCDF4')

    dataset = nc.Dataset(grid['LatLonOutputFileName'], 'w', format='NETCDF4')
    # with nc.Dataset(grid['LatLonOutputFileName'], 'w', format='NETCDF4') as dataset:
    dataset.createDimension('x', LI_grid_dims_CDO_format[1])
    dataset.createDimension('y', LI_grid_dims_CDO_format[0])
    dataset.createDimension('nv4', 4)
    

    lat_var = dataset.createVariable('lat', 'f4', ('y', 'x'))
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var.bounds = 'lat_bnds'
    lat_var[:, :] = LI_grid_center_lat

    lon_var = dataset.createVariable('lon', 'f4', ('y', 'x'))
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var.bounds = 'lon_bnds'
    lon_var[:, :] = LI_grid_center_lon

    # if len(LI_grid_dims_CDO_format) == 3:
    lat_bnds_var = dataset.createVariable('lat_bnds', 'f4', ('y', 'x', 'nv4'))
    lat_bnds_var.units = 'degrees_north'
    lat_bnds_var.standard_name = 'grid corner latitude'
    lat_bnds_var[:, :, :] = LI_grid_corner_lat_CDO_format

    lon_bnds_var = dataset.createVariable('lon_bnds', 'f4', ('y', 'x', 'nv4'))
    lon_bnds_var.units = 'degrees_east'
    lon_bnds_var.standard_name = 'grid corner longitude'
    lon_bnds_var[:, :, :] = LI_grid_corner_lon_CDO_format

    dummy_var = dataset.createVariable('dummy', 'i1', ('y', 'x'))
    dummy_var.units = '1'
    dummy_var.standard_name = 'dummy variable'
    dummy_var.coordinates = 'lon lat'
    dummy_var[:, :] = np.ones(LI_grid_dims_CDO_format, dtype=np.int8)
    dataset.close()
    
    if output_text:
        
        def write_array(file, array):
            try:
                reshaped = array.reshape(-1, 4)
            except ValueError:
                reshaped = np.append(array.reshape(-1), np.ones(4-(len(array) % 4))*np.nan).reshape(-1, 4)
                
            for i in reshaped:
                if any(np.isnan(i)):
                    i = list(i)
                    for j in range(len(i)):
                        if np.isnan(i[j]):
                            i[j] = ""
                    file.write(f"{i[0]:12.8f} {i[1]} {i[2]} {i[3]}\n")
                else:
                    file.write(f"{i[0]:12.8f} {i[1]:12.8f} {i[2]:12.8f} {i[3]:12.8f}\n")
        
        with open(grid['CDOOutputFileName'], 'w') as fileID:
            fileID.write('gridtype  = curvilinear\n')
            fileID.write(f'gridsize  = {nsize}\n')
            fileID.write(f'xsize  = {nx_centers}\n')
            fileID.write(f'ysize  = {ny_centers}\n')

            fileID.write('xvals  = \n')
            # [fileID.write(f"{a} {b} {c} {d}")]
            write_array(fileID, LI_grid_center_lon_CDO_format)
            fileID.write('\n')

            fileID.write('xbounds  = \n')
            write_array(fileID, LI_grid_corner_lon_CDO_format)
            # fileID.write(np.array2string(LI_grid_corner_lon_CDO_format))
            fileID.write('\n')

            fileID.write('yvals  = \n')
            write_array(fileID, LI_grid_center_lat_CDO_format)
            # fileID.write(np.array2string(LI_grid_center_lat_CDO_format))
            fileID.write('\n')

            fileID.write('ybounds  = \n')
            write_array(fileID, LI_grid_corner_lat_CDO_format)
            # fileID.write(np.array2string(LI_grid_corner_lat_CDO_format))
            fileID.write('\n')

        return 1


def regrid(ice_sheet, output_grids=None, output_text=False):
        
    
    # check ice_sheet parameter input
    ice_sheet = ice_sheet.lower()
    if ice_sheet not in ('ais', 'gis'):
        raise ValueError('ice_sheet must be one of either AIS or GIS')
    
    # specify output grids to be calculated 
    if output_grids is None:
        if ice_sheet == 'ais':
            output_grids = [16, 8, 4]
        elif ice_sheet == 'gis':
            output_grids = [5, 1]
    elif isinstance(output_grids, int):
        output_grids = [output_grids]
    elif isinstance(output_grids, list):
        pass
    else:
        raise ValueError('output_grids must be an integer or a list of integers')

    if ice_sheet == 'ais':
        # Specify mapping information. This is EPSG 3031
        proj_info = {
            'earthradius': 6378137.0,
            'eccentricity': 0.081819190842621,
            'standard_parallel': -71.0,
            'longitude_rot': 0.0,
            'falseeasting': 3040000,
            'falsenorthing': 3040000
        }
    elif ice_sheet in ('gis', 'gris'):
        # EPSG 3413
        proj_info = {
            'earthradius': 6378137.0,
            'eccentricity': 0.081819190842621,
            'standard_parallel': 70.0,
            'longitude_rot': 315.0,
            'falseeasting': 720000,
            'falsenorthing': 3450000
        }

    # Specify output angle type (degrees or radians)
    # output_data_type = 'radians'
    output_data_type = 'degrees'

    # Specify various ISM grids at different resolution
    # output_grids = [16, 8, 4] if ice_sheet == 'ais' else [5, 1]
    nx_base = 6081 if ice_sheet == 'ais' else 1681
    ny_base = 6081 if ice_sheet == 'ais' else 2881

    grids = []
    for r in output_grids:
        nx = (nx_base - 1) // r + 1
        ny = (ny_base - 1) // r + 1
        if isinstance(nx, int) and isinstance(ny, int):
            grid = {
                'dx': r * 1000.0,
                'dy': r * 1000.0,
                'nx_centers': (nx_base - 1) // r + 1,
                'ny_centers': (ny_base - 1) // r + 1,
                'LatLonOutputFileName': f'grid_ISMIP6_{ice_sheet}_{r*1000}m.nc',
                'CDOOutputFileName': f'grid_ISMIP6_{ice_sheet}_{r*1000}m.txt',
                'xyOutputFileName': f'xy_ISMIP6_{ice_sheet}_{r*1000}m.nc'
            }
            grids.append(grid)
        else:
            print(f'Warning: resolution {r} km is not commensurable, skipped.')

    # Create grids and write out
    for grid in grids:
        success = generate_CDO_files_nc(grid, proj_info, output_data_type, output_text=output_text)
    return success

regrid('AIS', output_text=True)
regrid('GIS', output_text=True)