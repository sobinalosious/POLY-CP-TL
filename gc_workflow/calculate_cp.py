import pandas as pd
import sys

# Define group contributions based on Satoh values in J/mol·K
group_cp_satoh = {
    "CH3": 35.54, "CH2": 25.35, "CH": 15.5, "C": 6.2, "O": 25.98, "C=O": 30.71, "COO": 53.37, "COOH": 50,
    "=CH2": 22.6, "=CH": 27.7, "=C<": 10.5, "-F": 32.81, "-Cl": 27.1, "-Br": 78.24, "-I": 140.45, "-OH": 26.41,
    "-NO2": 41.9, "-NH2": 20.94, "-NH-": 14.24, ">N-": 17.1, "-S-": 24, "-SH": 46.8, "-SO2-": 49.72,
    "1-substituted benzene": 92, "2-substituted benzene": 85, "3-substituted benzene": 80,
    "4-substituted benzene": 80, "5-substituted benzene": 80, "6-substituted benzene": 80,
    "CH2 (5 ring)": 19.9, "CH2 (6 ring)": 18.0, "CN": 25, "CONH": 54,
    # New groups with Cp set to zero
    "Si": 0, "B": 0, "N": 17.1, "H": 0, "P": 0, "Sn": 0, "Na": 0, "Li": 0, 
    "Ge": 0, "Se": 0, "K": 0, "Fe": 0, "Co": 0, "Ni": 0, "Ca": 0, 
    "Cd": 0, "Pb": 0, "Zn": 0, "Te": 0, "As": 0
}


# Define atomic masses (g/mol) for the elements involved
atomic_masses = {
    "C": 12.01, "H": 1.008, "O": 16.00, "N": 14.01, "F": 19.00, "Cl": 35.45, 
    "Br": 79.90, "I": 126.90, "S": 32.07, "Si": 28.09, "B": 10.81, "P": 30.97,
    "Sn": 118.71, "Na": 22.99, "Li": 6.94, "Ge": 72.63, "Se": 78.96, "K": 39.10,
    "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Ca": 40.08, "Cd": 112.41, 
    "Pb": 207.2, "Zn": 65.38, "Te": 127.6, "As": 74.92
}

# Define the molar masses of each functional group based on atomic composition
group_molar_mass = {
    "CH3": atomic_masses["C"] + 3 * atomic_masses["H"],
    "CH2": atomic_masses["C"] + 2 * atomic_masses["H"],
    "CH": atomic_masses["C"] + atomic_masses["H"],
    "C": atomic_masses["C"],
    "O": atomic_masses["O"],
    "C=O": atomic_masses["C"] + atomic_masses["O"],
    "COO": atomic_masses["C"] + 2 * atomic_masses["O"],
    "COOH": atomic_masses["C"] + 2 * atomic_masses["O"] + atomic_masses["H"],
    "=CH2": atomic_masses["C"] + 2 * atomic_masses["H"],
    "=CH": atomic_masses["C"] + atomic_masses["H"],
    "=C<": atomic_masses["C"],
    "-F": atomic_masses["F"],
    "-Cl": atomic_masses["Cl"],
    "-Br": atomic_masses["Br"],
    "-I": atomic_masses["I"],
    "-OH": atomic_masses["O"] + atomic_masses["H"],
    "-NO2": atomic_masses["N"] + 2 * atomic_masses["O"],
    "-NH2": atomic_masses["N"] + 2 * atomic_masses["H"],
    "-NH-": atomic_masses["N"] + atomic_masses["H"],
    ">N-": atomic_masses["N"],
    "-S-": atomic_masses["S"],
    "-SH": atomic_masses["S"] + atomic_masses["H"],
    "-SO2-": atomic_masses["S"] + 2 * atomic_masses["O"],
    "1-substituted benzene": 6 * atomic_masses["C"] + 5 * atomic_masses["H"],
    "2-substituted benzene": 6 * atomic_masses["C"] + 4 * atomic_masses["H"],
    "3-substituted benzene": 6 * atomic_masses["C"] + 3 * atomic_masses["H"],
    "4-substituted benzene": 6 * atomic_masses["C"] + 2 * atomic_masses["H"],
    "5-substituted benzene": 6 * atomic_masses["C"] + 1 * atomic_masses["H"],
    "6-substituted benzene": 6 * atomic_masses["C"] + atomic_masses["H"],
    "CH2 (5 ring)": 5 * atomic_masses["C"] + 4 * atomic_masses["H"],
    "CH2 (6 ring)": 6 * atomic_masses["C"] + 5 * atomic_masses["H"],
    "CN": atomic_masses["C"] + atomic_masses["N"],
    "CONH": atomic_masses["C"] + atomic_masses["O"] + atomic_masses["N"] + atomic_masses["H"],
    # New groups with atomic masses included
    "Si": atomic_masses["Si"], "B": atomic_masses["B"], "N": atomic_masses["N"], 
    "H": atomic_masses["H"], "P": atomic_masses["P"], "Sn": atomic_masses["Sn"], 
    "Na": atomic_masses["Na"], "Li": atomic_masses["Li"], "Ge": atomic_masses["Ge"],
    "Se": atomic_masses["Se"], "K": atomic_masses["K"], "Fe": atomic_masses["Fe"],
    "Co": atomic_masses["Co"], "Ni": atomic_masses["Ni"], "Ca": atomic_masses["Ca"],
    "Cd": atomic_masses["Cd"], "Pb": atomic_masses["Pb"], "Zn": atomic_masses["Zn"],
    "Te": atomic_masses["Te"], "As": atomic_masses["As"]
}


# Load the data from GROUPS.csv

file_path = sys.argv[1]  # Update this path if the file is located elsewhere
groups_df = pd.read_csv(file_path)

# Identify groups with zero Cp contribution
zero_cp_groups = {group for group, cp in group_cp_satoh.items() if cp == 0}

# Define the list of functional groups in the expected order
functional_group_counts = list(group_cp_satoh.keys())

# Function to determine if polymer has no zero Cp contribution groups
def has_no_zero_cp_group(row):
    for i, group in enumerate(functional_group_counts, start=1):  # Start=2 to skip PID and SMILES
        count = row[i]
        if group in zero_cp_groups and count > 0:
            return False  # Polymer has at least one functional group with zero Cp contribution
    return True  # No zero Cp contribution groups found in this polymer

# Apply the function and create a new column "No Zero Cp Group"
groups_df['No Zero Cp Group'] = groups_df.apply(has_no_zero_cp_group, axis=1)


# Open a file to write the calculation steps
calculation_file = open("calculation.txt", "w")

# Function to calculate Cp in J/(kg·K) and molar mass based on functional groups
def calculate_cp_and_molar_mass(row):
    cp_molar = 0.0
    molar_mass = 0.0
    calculation_file.write(f"Calculations for  (SMILES: {row['SMILES']}):\n")
    
    for i, group in enumerate(functional_group_counts, start=1):
        count = row[i]
        if count > 0:
            contribution_cp = group_cp_satoh.get(group, 0) * count
            cp_molar += contribution_cp
            contribution_mass = group_molar_mass.get(group, 0) * count
            molar_mass += contribution_mass
            calculation_file.write(
                f"  Group: {group}, Count: {count}, "
                f"Contribution to Cp: {contribution_cp:.2f} J/(mol·K), "
                f"Contribution to Molar Mass: {contribution_mass:.2f} g/mol\n"
            )
    
    if molar_mass == 0:
        calculation_file.write(f"  ❗ Skipped due to zero molar mass (invalid or unmatched groups)\n\n")
        return float('nan'), float('nan')  # or 0.0, 0.0 if you prefer

    cp_si = round(cp_molar / (molar_mass / 1000), 2)
    molar_mass = round(molar_mass, 2)
    calculation_file.write(
        f"Total Cp: {cp_si} J/(kg·K), Molar Mass: {molar_mass} g/mol\n\n"
    )
    
    return cp_si, molar_mass


# Apply the calculation to each polymer and save results
groups_df[['Cp', 'Molar Mass (g/mol)']] = groups_df.apply(
    lambda row: pd.Series(calculate_cp_and_molar_mass(row)), axis=1
)

# Close the calculation file
calculation_file.close()

# Save the results to CP_RESULTS_SI.csv in SI units
output_file_path = sys.argv[2]

# Keep only polymers with no zero-Cp groups
filtered_df = groups_df[groups_df['No Zero Cp Group']]

filtered_df[['SMILES', 'Cp', 'Molar Mass (g/mol)']].to_csv(
    output_file_path,
    index=False
)


print("Calculation complete.")
