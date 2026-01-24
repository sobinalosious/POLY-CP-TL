import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import re
import sys

# Read SMILES data from CSV file
smiles_file = sys.argv[1]
smiles_data = pd.read_csv(smiles_file, header=0, encoding='ISO-8859-1')

# Initialize list to store results
results = []

# Iterate over each row in the SMILES data
for index, row in smiles_data.iterrows():
    #pid = row[0]  # First column (PID)
    psmiles = row["SMILES"]
  # Second column (SMILES)

    # Convert SMILES to RDKit molecule with explicit hydrogens
    mol = Chem.MolFromSmiles(psmiles)
    mol = Chem.AddHs(mol)

    # Calculate molecular formula using RDKit
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    # Initialize dictionary to store counts of functional groups
    functional_group_counts = {
        "CH3": 0, "CH2": 0, "CH": 0, "C": 0, "O": 0, "C=O": 0, "COO": 0, "COOH": 0,
        "=CH2": 0, "=CH": 0, "=C<": 0, "-F": 0, "-Cl": 0, "-Br": 0, "-I": 0, "-OH": 0,
        "-NO2": 0, "-NH2": 0, "-NH-": 0, ">N-": 0, "-S-": 0, "-SH": 0, "-SO2-": 0,
        "1-substituted benzene": 0, "2-substituted benzene": 0, "3-substituted benzene": 0,
        "4-substituted benzene": 0, "5-substituted benzene": 0, "6-substituted benzene": 0,
        "CH2 (5 ring)": 0, "CH2 (6 ring)": 0, "CN": 0, "CONH": 0
    }

    # Track oxygen atoms in COO, COOH, or OH groups to avoid double-counting
    functional_group_oxygens = set()

    # Pattern to identify benzene rings and count external connections for substitution classification
    benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
    benzene_rings = mol.GetSubstructMatches(benzene_pattern)
    for ring in benzene_rings:
        external_connections = set()
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in ring and neighbor.GetAtomicNum() != 1:
                    external_connections.add(neighbor.GetIdx())
        substitution_count = len(external_connections)
        if substitution_count > 0:
            functional_group_counts[f"{substitution_count}-substituted benzene"] += 1

    # Count hydrogens attached to benzene carbons
    benzene_hydrogens = sum(
        sum(1 for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if neighbor.GetAtomicNum() == 1)
        for ring in benzene_rings for atom_idx in ring
    )

    # Identify and count CH2 in 5- and 6-membered rings
    ch2_5_ring_pattern = Chem.MolFromSmarts("[R2&R5]C([H])([H])")
    ch2_6_ring_pattern = Chem.MolFromSmarts("[R2&R6]C([H])([H])")
    functional_group_counts["CH2 (5 ring)"] += len(mol.GetSubstructMatches(ch2_5_ring_pattern))
    functional_group_counts["CH2 (6 ring)"] += len(mol.GetSubstructMatches(ch2_6_ring_pattern))

    # Identify and count other functional groups (continued from original code)
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()

        # Carbon atoms (classify based on bonds and neighbors)
        if atomic_num == 6:
            if any(atom.GetIdx() in ring for ring in benzene_rings):
                continue
            num_hydrogens = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
            double_bonded_carbons = [
                neighbor for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() == 6 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            ]

            # Classify unsaturated carbons
            if len(double_bonded_carbons) == 1:
                if num_hydrogens == 2:
                    functional_group_counts["=CH2"] += 1
                elif num_hydrogens == 1:
                    functional_group_counts["=CH"] += 1
                elif num_hydrogens == 0:
                    functional_group_counts["=C<"] += 1
            # Classify other carbon types
            else:
                if num_hydrogens == 3:
                    functional_group_counts["CH3"] += 1
                elif num_hydrogens == 2:
                    functional_group_counts["CH2"] += 1
                elif num_hydrogens == 1:
                    functional_group_counts["CH"] += 1
                else:
                    is_carbonyl, is_coo, is_cooh = False, False, False
                    is_carbonyl = any(
                        neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                        for neighbor in atom.GetNeighbors()
                    )
                    if is_carbonyl:
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                                is_coo = any(sub_neighbor.GetAtomicNum() == 6 for sub_neighbor in neighbor.GetNeighbors() if sub_neighbor.GetIdx() != atom.GetIdx())
                                is_cooh = any(sub_neighbor.GetAtomicNum() == 1 for sub_neighbor in neighbor.GetNeighbors() if sub_neighbor.GetIdx() != atom.GetIdx())
                                functional_group_oxygens.add(neighbor.GetIdx())
                                break
                    if is_cooh:
                        functional_group_counts["COOH"] += 1
                    elif is_coo:
                        functional_group_counts["COO"] += 1
                    elif is_carbonyl:
                        functional_group_counts["C=O"] += 1
                    else:
                        functional_group_counts["C"] += 1

        # Oxygen atoms
        elif atomic_num == 8:
            if atom.GetIdx() in functional_group_oxygens:
                continue
            if any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()):
                functional_group_counts["-OH"] += 1
            else:
                is_single_bonded = all(
                    mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() != Chem.BondType.DOUBLE
                    for neighbor in atom.GetNeighbors()
                )
                if is_single_bonded:
                    functional_group_counts["O"] += 1

        # Halogens
        elif atomic_num in {9, 17, 35, 53}:
            halogen_map = {9: "-F", 17: "-Cl", 35: "-Br", 53: "-I"}
            functional_group_counts[halogen_map[atomic_num]] += 1

        # Nitrogen and Sulfur atoms
        elif atomic_num == 7:
            num_hydrogens = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)
            if num_hydrogens == 2:
                functional_group_counts["-NH2"] += 1
            elif num_hydrogens == 1:
                functional_group_counts["-NH-"] += 1
            elif num_hydrogens == 0:
                if len(atom.GetNeighbors()) == 3:
                    functional_group_counts[">N-"] += 1
                elif all(neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for neighbor in atom.GetNeighbors()):
                    functional_group_counts["-NO2"] += 1
        elif atomic_num == 16:
            if any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()):
                functional_group_counts["-SH"] += 1
            elif sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE) == 2:
                functional_group_counts["-SO2-"] += 1
            else:
                functional_group_counts["-S-"] += 1

        # Count CN and CONH groups
        if atomic_num == 6:
            if any(neighbor.GetAtomicNum() == 7 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() in [Chem.BondType.DOUBLE,Chem.BondType.TRIPLE]for neighbor in atom.GetNeighbors()):
                functional_group_counts["CN"] += 1
            if any(neighbor.GetAtomicNum() == 7 and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE for neighbor in atom.GetNeighbors()):
                for n_neighbor in atom.GetNeighbors():
                    if n_neighbor.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(n_neighbor.GetIdx(), atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                        functional_group_counts["CONH"] += 1

    # Construct molecular formula from functional group counts
    
    
    #coo_count = functional_group_counts["COO"]
    #if coo_count > 0:
        #functional_group_counts["O"] -= min(coo_count, functional_group_counts["O"])
    # Apply logic to adjust carbon count based on CN presence
    #cn_count = functional_group_counts["CN"]
    #if cn_count > 0:
    # Subtract 1 carbon per CN group, ensuring the count doesn't go below zero
        #functional_group_counts["C"] -= min(cn_count, functional_group_counts["C"])
        
        
    formula_counts = {"C": 0, "H": 0, "O": 0, "N": 0, "S": 0, "F": 0, "Cl": 0, "Br": 0, "I": 0}
    # Summing element counts from functional group occurrences
    formula_counts["C"] += functional_group_counts["CH3"] + functional_group_counts["CH2"] + functional_group_counts["CH"] + functional_group_counts["C"] + \
                            functional_group_counts["=CH2"] + functional_group_counts["=CH"] + functional_group_counts["=C<"] + \
                            functional_group_counts["C=O"] + functional_group_counts["COO"] + functional_group_counts["COOH"] + \
                            functional_group_counts["CH2 (5 ring)"] + functional_group_counts["CH2 (6 ring)"] + len(benzene_rings) * 6 + \
                            functional_group_counts["CN"] + functional_group_counts["CONH"]
    formula_counts["H"] += functional_group_counts["CH3"] * 3 + functional_group_counts["CH2"] * 2 + functional_group_counts["CH"] + \
                            functional_group_counts["=CH2"] * 2 + functional_group_counts["=CH"] + functional_group_counts["-OH"] + \
                            functional_group_counts["-NH2"] * 2 + functional_group_counts["-NH-"] + functional_group_counts["-SH"] + \
                            functional_group_counts["COOH"] + functional_group_counts["CH2 (5 ring)"] * 2 + \
                            functional_group_counts["CH2 (6 ring)"] * 2 + benzene_hydrogens + \
                            functional_group_counts["CONH"] * 1
    formula_counts["O"] += functional_group_counts["O"] + functional_group_counts["C=O"] + functional_group_counts["COO"] * 2 + \
                            functional_group_counts["COOH"] * 2 + functional_group_counts["-OH"] + functional_group_counts["-NO2"] * 2 + \
                            functional_group_counts["-SO2-"] * 2 + functional_group_counts["CONH"]
    formula_counts["N"] += functional_group_counts["-NH2"] + functional_group_counts["-NH-"] + functional_group_counts[">N-"] + \
                            functional_group_counts["-NO2"] + functional_group_counts["CN"] + functional_group_counts["CONH"]
    formula_counts["S"] += functional_group_counts["-S-"] + functional_group_counts["-SH"] + functional_group_counts["-SO2-"]
    formula_counts["F"] += functional_group_counts["-F"]
    formula_counts["Cl"] += functional_group_counts["-Cl"]
    formula_counts["Br"] += functional_group_counts["-Br"]
    formula_counts["I"] += functional_group_counts["-I"]

    # Construct molecular formula from counts
    constructed_formula = "".join(f"{element}{count if count > 1 else ''}" for element, count in formula_counts.items() if count > 0)

    # Compare RDKit and constructed formulas
    rdkit_formula_dict = {k: int(v) if v else 1 for k, v in re.findall(r'([A-Z][a-z]*)(\d*)', molecular_formula)}
    constructed_formula_dict = {k: int(v) if v else 1 for k, v in re.findall(r'([A-Z][a-z]*)(\d*)', constructed_formula)}
    all_elements = set(rdkit_formula_dict.keys()).union(set(constructed_formula_dict.keys()))

    # Calculate total difference and also create a difference description
    element_differences = []
    total_difference = 0
        # Adjust functional groups based on the difference
    for element in all_elements:
        rdkit_count = rdkit_formula_dict.get(element, 0)
        constructed_count = constructed_formula_dict.get(element, 0)
        difference = rdkit_count - constructed_count
        if difference > 0:
            # Add required atoms to match RDKit formula
            functional_group_counts[element] = functional_group_counts.get(element, 0) + difference
        elif difference < 0:
            # Remove excess atoms to match RDKit formula
            available = functional_group_counts.get(element, 0)
            functional_group_counts[element] = max(available + difference, 0)  # Ensure non-negative counts

    # Join the element differences into a string
    element_differences_str = ", ".join(element_differences)

    # Append results for current SMILES
    result = {
        "SMILES": psmiles,
        **{f'"{key}"': value for key, value in functional_group_counts.items()},
        "formula_difference": total_difference,
        "element_differences": element_differences_str,
        #"rdkit_formula": molecular_formula,
        #"calculated_formula": constructed_formula
     
    }
    results.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
results_df = results_df.fillna(0)

# Save results to a CSV file
group_file = sys.argv[2]
results_df.to_csv(group_file, index=False)

