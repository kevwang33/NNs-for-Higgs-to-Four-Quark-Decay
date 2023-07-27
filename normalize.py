import numpy as np
import h5py

#read in the data file to be normalized
f = h5py.File('data_HWWqqqq.hdf5', 'a')
fjc_data = f["jet components"][:]
jet_data = f['jet attributes'][:]



normalized_data = fjc_data.copy()
updated_jet = jet_data.copy()


def handle_period(number):
    """
    this function handles the 
    periodicity of the jet eta 
    and phi and keeps their 
    value between pi and -pi
    """
    while abs(number)>np.pi:
        if number > np.pi:
            number = number - 2*np.pi
        if number < -np.pi:
            number = number +2*np.pi

    return number

def update_jet(four_vector):
    """
    this function updates the jet PT and mass 
    from MeV to GeV
    """
    four_vector[0] = four_vector[0]*10**-3
    four_vector[3] = four_vector[3]*10**-3
  
    return four_vector


for i in range(len(fjc_data)):
    for j in range(30):
        if normalized_data[i][j].tolist() != [0,0,0,0]:
            normalized_data[i][j] = [fjc_data[i][j][0]/jet_data[i][0],handle_period(fjc_data[i][j][1]-jet_data[i][1]), handle_period(fjc_data[i][j][2]-jet_data[i][2]), fjc_data[i][j][3]]
        else:
            continue
    updated_jet[i] = update_jet(jet_data[i])



dset1 = f.create_dataset("normalized jet components", data = normalized_data)
dset2 = f.create_dataset("updated jet attributes", data = updated_jet)
f.close()
