import numpy as np
import linecache
import os
import sys
import argparse
import shutil
import glob

from top_embedding import SimplicialComplex_laplacian


def get_data_from_protein_PDB(
    protein_pdb_file, consider_ele=['C', 'N', 'O', 'S']
):
    ####
    selected_ele, selected_xyz = [], []
    for i, line in enumerate(open(protein_pdb_file)):
        if line.startswith('ATOM'):
            line_ele = line.strip().split()
            ele = line_ele[2][0]  # first str is used
            if ele in consider_ele:
                selected_ele.append(ele)

                # based on '.' symbol to split
                ele_v2 = line.strip().split('.')
                x = ele_v2[0][-4::].strip() + '.' + ele_v2[1][0:3].strip()
                y = ele_v2[1][-4::].strip() + '.' + ele_v2[2][0:3].strip()
                z = ele_v2[2][-4::].strip() + '.' + ele_v2[3][0:3].strip()
                selected_xyz.append([float(x), float(y), float(z)])
    return np.array(selected_ele), np.array(selected_xyz) 


def get_data_from_ligand_sdf(
    ligand_sdf_file,
    consider_ele=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
):  
    # get line number, atom num
    atom_num = int(linecache.getline(ligand_sdf_file, 4)[0:3].strip())
    start_linenum = 4
    end_linenum = 4 + atom_num

    # read file
    selected_ele, selected_xyz = [], []
    for i in range(start_linenum+1, end_linenum+1):  # lineache start at 1
        line = linecache.getline(ligand_sdf_file, i)
        line_ele = line.strip().split()
        ele = line_ele[3]
        if ele in consider_ele:
            selected_ele.append(ele)
            selected_xyz.append([float(xx) for xx in line_ele[0:3]])
    return np.array(selected_ele), np.array(selected_xyz)


def get_data_from_ligand_ensemble_PDB(
        protein_pdb_file,  # ligand form the complex PDB file
        consider_ele=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
):
    ####
    selected_ele, selected_xyz = [], []
    for i, line in enumerate(open(protein_pdb_file)):
        if line.startswith('HETATM') and ('4WI A' in line):
            line_ele = line.strip().split()
            ele = line_ele[-1]
            if ele in consider_ele:
                selected_ele.append(ele)

                # based on '.' symbol to split
                ele_v2 = line.strip().split('.')
                x = ele_v2[0][-4::].strip() + '.' + ele_v2[1][0:3].strip()
                y = ele_v2[1][-4::].strip() + '.' + ele_v2[2][0:3].strip()
                z = ele_v2[2][-4::].strip() + '.' + ele_v2[3][0:3].strip()
                selected_xyz.append([float(x), float(y), float(z)])
    return np.array(selected_ele), np.array(selected_xyz) 



def get_data_from_ligand_PDB(
        protein_pdb_file,  # ligand form the complex PDB file
        consider_ele=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
):
    ####
    selected_ele, selected_xyz = [], []
    for i, line in enumerate(open(protein_pdb_file)):
        if line.startswith('ATOM'):
            line_ele = line.strip().split()
            ele = line_ele[-1]
            if ele in consider_ele:
                selected_ele.append(ele)

                # based on '.' symbol to split
                ele_v2 = line.strip().split('.')
                x = ele_v2[0][-4::].strip() + '.' + ele_v2[1][0:3].strip()
                y = ele_v2[1][-4::].strip() + '.' + ele_v2[2][0:3].strip()
                z = ele_v2[2][-4::].strip() + '.' + ele_v2[3][0:3].strip()
                selected_xyz.append([float(x), float(y), float(z)])
    return np.array(selected_ele), np.array(selected_xyz) 


def get_data_from_ligand_mol2(
        ligand_mol2_file,
        consider_ele=['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    ):
    # 
    selected_ele, selected_xyz = [], []
    flag = 0
    for i, line in enumerate(open(ligand_mol2_file)):
        if line.startswith('@<TRIPOS>ATOM'):
            flag = 1
            continue

        if line.startswith('@<TRIPOS>BOND'):
            flag = 0
            break
        
        if flag == 1:
            line_ele = line.strip().split()
            ele = line_ele[5].split('.')[0]
            if ele in consider_ele:
                selected_ele.append(ele)
                selected_xyz.append([float(xx) for xx in line_ele[2:5]])
    return np.array(selected_ele), np.array(selected_xyz)


def selected_target_combination(scheme_name='ele_scheme_1') -> tuple:
    # protein-ligand combination schemes
    # scheme_1-protein: single heavy atom, double heavy atom, all heavy atom.
    # scheme_1-ligand: top 4 from element distribution; 2-combinations of top4 elements; group-5, 6, 7 in periodic table, ('O', 'S') is group-6 repetition; all heavy atom
    # scheme_2-protein: all single heavy atoms
    # scheme_2-ligand: all single elements
    protein_combinations = {
        'ele_scheme_1': [('C',), ('N',), ('O',), ('S',), ('C', 'N'), ('C', 'O'), ('C', 'S'), ('N', 'O'), ('N', 'S'), ('O', 'S'), ('C', 'N', 'O', 'S')],
        'scheme_2': [('C',), ('N',), ('O',), ('S',)],
    }
    ligand_combinations = {
        'ele_scheme_1': [('C',), ('N',), ('O',), ('S',), ('C', 'N'), ('C', 'O'), ('C', 'S'), ('N', 'O'), ('N', 'S'), ('O', 'S'), ('N', 'P'), ('F', 'Cl', 'Br', 'I'), ('C', 'O', 'N', 'S', 'F', 'P', 'Cl', 'Br', 'I')], 
        'scheme_2': [('H',), ('C',), ('N',), ('O',), ('F',), ('P',), ('S',), ('Cl'), ('Br'), ('I',)],
    }
    return protein_combinations[scheme_name], ligand_combinations[scheme_name]


def generate_distance_matrix(protein_ele, protein_xyz, ligand_ele, ligand_xyz, cutoff=100):
    ligand_len = len(ligand_ele)
    protein_len = len(protein_ele)
    all_len = protein_len + ligand_len

    if ligand_len > 0 and protein_len > 0:
        xyz = np.vstack([protein_xyz, ligand_xyz])
        distance_matrix = np.zeros([protein_len + ligand_len]*2)
        for i, i_xyz in enumerate(xyz):
            for j, j_xyz in enumerate(xyz):
                if j <= i:
                    continue
                if (i<protein_len and j<protein_len) or (i>=protein_len and j>=protein_len):
                    distance_matrix[i, j] = cutoff*10000
                    distance_matrix[j, i] = cutoff*10000
                else:
                    distance_matrix[i, j] = np.linalg.norm(i_xyz-j_xyz)
                    distance_matrix[j, i] = distance_matrix[i, j]
    elif (ligand_len==0 and protein_len>0) or (ligand_len>0 and protein_len==0) or (
        ligand_len == 0 and protein_len == 0
    ):
        distance_matrix = np.ones([all_len, all_len]) * cutoff * 10000
        np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def generate_lap_features(
    protein_feature_folder,
    protein_pdb,
    ligand_mol2,
    consider_field,
    ph_cutoff,
    ph_startdis,
    ele_scheme,
    ligand_type,
):
    # all protein and all ligand
    raw_protein_ele, raw_protein_xyz = get_data_from_protein_PDB(
        protein_pdb_file=protein_pdb,
        consider_ele=['C', 'N', 'O', 'S'],
    )
    if ligand_type == 'pdb':
        raw_ligand_ele, raw_ligand_xyz = get_data_from_ligand_PDB(
            ligand_mol2,
            ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        )
    elif ligand_type == 'sdf':
        raw_ligand_ele, raw_ligand_xyz = get_data_from_ligand_sdf(
            ligand_mol2,
            ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        )
    elif ligand_type == 'ensamble_pdb':
        raw_ligand_ele, raw_ligand_xyz = get_data_from_ligand_ensemble_PDB(
            ligand_mol2,
            ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        )

    # reserve consider_field spherical area around the ligand
    protein_indices = []
    flag = 0
    for i_portein, protein_xyz in enumerate(raw_protein_xyz):
        for i_ligand, ligand_xyz in enumerate(raw_ligand_xyz):
            if np.linalg.norm(protein_xyz-ligand_xyz) <= consider_field:
                flag = 1
                break
            else:
                flag = 0
                continue
        if flag == 1:
            protein_indices.append(i_portein)
    considered_protein_ele = raw_protein_ele[protein_indices]
    considered_protein_xyz = raw_protein_xyz[protein_indices]
    print(f'protein size: {len(considered_protein_ele)}')

    #
    protein_ele_sets, ligand_ele_sets = selected_target_combination(scheme_name=ele_scheme)
    feature_array = np.zeros([6, 100, 143])
    element_combin_count = 0
    for i, protein_ele in enumerate(protein_ele_sets):
        selected_protein_indices = [ele in protein_ele for ele in considered_protein_ele]
        selected_protein_ele = considered_protein_ele[selected_protein_indices]
        selected_protein_xyz = considered_protein_xyz[selected_protein_indices]
        for j, ligand_ele in enumerate(ligand_ele_sets):
            selected_ligand_indices = [ele in ligand_ele for ele in raw_ligand_ele]
            selected_ligand_ele = raw_ligand_ele[selected_ligand_indices]
            selected_ligand_xyz = raw_ligand_xyz[selected_ligand_indices]

            # make and save distance matrix file
            distance_matrix = generate_distance_matrix(selected_protein_ele, selected_protein_xyz, selected_ligand_ele, selected_ligand_xyz, ph_cutoff)
            
            # running laplacian program
            protein_len = len(selected_protein_ele)
            ligand_len = len(selected_ligand_ele)
            if (ligand_len > 0 and protein_len > 0):
                scpl = SimplicialComplex_laplacian.SimplicialComplexLaplacian()
                all_laplacian_features = scpl.persistent_simplicialComplex_laplacian_dim0(
                    input_data=distance_matrix,
                    is_distance_matrix=True,
                    max_dim=0,
                    filtration=np.round(np.arange(ph_startdis, ph_cutoff, 0.1), 2),
                    print_by_step=True,
                )
                for filtration_n, data_l in enumerate(all_laplacian_features):
                    statistic_val = SimplicialComplex_laplacian.statistic_eigvalues(data_l[0])
                    statistic_feature = [
                        statistic_val.count_zero,
                        statistic_val.max,
                        statistic_val.sum,
                        statistic_val.nonzero_mean,
                        statistic_val.nonzero_std,
                        statistic_val.nonzero_min,
                    ]
                    feature_array[:, filtration_n, element_combin_count] = statistic_feature
            element_combin_count += 1
    
    # save
    protein_name_temp = os.path.split(protein_pdb)[-1].split('.pdb')[0]
    protein_name = protein_name_temp.split('_protein')[0] if '_protein' in protein_name_temp else protein_name_temp
    np.save(os.path.join(protein_feature_folder, f'{protein_name}.npy'), feature_array, allow_pickle=True)

    return None


def parse_args(args):
    parser = argparse.ArgumentParser(description="generate topological features")
    parser.add_argument('--protein_feature_folder', default='./dd', type=str)
    parser.add_argument('--protein_pdb', default='ele_scheme_1-protein_only_ph_vr-10.npy', type=str)
    parser.add_argument('--ligand_mol2', default='ele_scheme_1-protein_only_ph_vr-10.npy', type=str)
    parser.add_argument('--consider_field', default=20, type=float)
    parser.add_argument('--ph_cutoff', default=10, type=float)
    parser.add_argument('--ph_startdis', default=0, type=float)
    parser.add_argument('--ele_scheme', default='ele_scheme_1', type=str)
    parser.add_argument('--ligand_file_type', default='pdb', type=str, help='ensamble_pdb for mutant pdb')
    args = parser.parse_args()
    return args


def main():
    args = parse_args(sys.argv[1:])
    generate_lap_features(
        protein_feature_folder=args.protein_feature_folder,
        protein_pdb=args.protein_pdb,
        ligand_mol2=args.ligand_mol2,
        consider_field=args.consider_field,
        ph_cutoff=args.ph_cutoff,
        ph_startdis=args.ph_startdis,
        ele_scheme=args.ele_scheme,
        ligand_type=args.ligand_file_type,
    )
    return None


if __name__ == "__main__":
    # test code
    # python ./main_bindingdb_protein_ligand_complex_feature.py --protein_feature_folder /mnt/research/guowei-search.2/DongChen/BindingDB_docked_features_212_20 --protein_pdb /mnt/research/guowei-search.2/DongChen/BindingDB_docked_data/3C3U_C2U/3C3U.pdb --ligand_mol2 /mnt/research/guowei-search.2/DongChen/BindingDB_docked_data/3C3U_C2U/3C3U_ligand0.mol2 --ph_startdis 0 --ph_cutoff 10 --consider_field 20 
    main()
    # test_function()
