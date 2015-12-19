from pymatgen import MPRester, periodic_table
import itertools
import csv

# mpr = MPRester(MAPI_KEY) # You have to register with Materials Project to receive an API

# There are 103 elements in pymatgen's list, giving C(103, 2) = 5253 binary systems
allBinaries = itertools.combinations(periodic_table.all_symbols(), 2)  # Create list of all binary systems

with MPRester() as m:
    with open('bandgap_energy_densityDFT.csv', 'wb') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter=',')
        for system in allBinaries:
            results = m.get_data(system[0] + '-' + system[1],
                                 data_type='vasp')  # Download DFT data for each binary system
            # print results
            for material in results:  # We will receive many compounds within each binary system
                if material['e_above_hull'] < 1e-6:  # Check if this compound is thermodynamically stable
                    dat = []
                    dat.append(material['pretty_formula'])
                    dat.append(str(material['band_gap']))
                    dat.append(str(material['formation_energy_per_atom']))
                    dat.append(str(material['density']))
                    csv_writer.writerow(dat)


csvfile.close()
