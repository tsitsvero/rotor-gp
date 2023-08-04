

#Hydrogen: 
ind_H_1 = sorted([117, 119, 123, 125, 135, 137, 116, 118, 122, 124, 134, 136]) 
ind_H_2 = sorted([115, 121, 127, 129, 133, 139, 132, 138, 126, 128, 114, 120] )
ind_H_3 = sorted([130, 131, 140, 141, 142, 143, 144, 145, 164, 165, 166, 167, 192, 193, 206, 207, 250, 251] )
ind_H_4 = sorted([146, 148, 150, 147, 149, 151, 152, 154, 156, 153, 155, 157, 158, 160, 162, 159, 161, 163, 168, 170, 172, 169, 171, 173, 174, 176, 178, 175, 177, 179, 180, 182, 184, 181, 183, 185, 186, 188, 190, 187, 189, 191, 194, 196, 198, 195, 197, 199, 200, 202, 204, 201, 203, 205, 208, 210, 212, 209, 211, 213, 214, 216, 218, 215, 217, 219, 220, 222, 224, 221, 223, 225, 226, 228, 230, 227, 229, 231, 232, 234, 236, 233, 235, 237, 238, 240, 242, 239, 241, 243, 244, 246, 248, 245, 247, 249, 252, 254, 256, 253, 255, 257, 258, 260, 262, 259, 261, 263] )
#Carbon: 
ind_C_1 = sorted([6, 8, 10, 7, 9, 11] )
ind_C_2 = sorted([18, 26, 44, 19, 27, 45] )
ind_C_3 = sorted([24, 28, 16, 20, 42, 46, 17, 21, 25, 29, 43, 47] )
ind_C_4 = sorted([30, 34, 14, 22, 40, 48, 15, 23, 31, 35, 41, 49] )
ind_C_5 = sorted([12, 13, 32, 33, 38, 39] )
ind_C_6 = sorted([36, 50, 64, 37, 51, 65, 52, 54, 80, 53, 55, 81, 62, 74, 96, 63, 75, 97] )
ind_C_7 = sorted([56, 60, 57, 61, 68, 72, 69, 73, 66, 70, 67, 71, 78, 84, 79, 85, 58, 59, 88, 90, 89, 91, 86, 87, 92, 94, 93, 95, 76, 77, 82, 83, 98, 100, 99, 101] )
#Nytrogen: 
ind_N_1 = sorted( [102, 103, 104, 105, 106, 107] )
#Oxygen: 
ind_O_1 = sorted( [108, 109, 110, 111, 112, 113] )
#Silicon: 
ind_Si_1 = sorted( [0, 1, 2, 3, 4, 5] )

# len(ind_C_1 + ind_C_2 + ind_C_3 + ind_C_4 + ind_C_5 + ind_C_6 + ind_C_7) + len(ind_H_1 + ind_H_2 + ind_H_3 + ind_H_4)+ len(ind_N_1)+ len(ind_O_1)+ len(ind_Si_1)



def prepare_data(hparams, soap_params):
    

        # print("Starting script...")
        import time
        start_time = time.time()

        # export R=0; python ./train_and_run.py 2>&1 

        import ase
        from ase import io
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        import sys
        sys.path.append("../../fande") 
        # sys.path.append("..") 
        import fande

        from datetime import datetime

        os.environ['VASP_PP_PATH'] = "/home/qklmn/repos/pseudos"

        def make_calc_dir():
                if os.getcwd()[-4:] != 'code':
                        os.chdir('../../../code')
                dir_name = f'{datetime.now().strftime("%Y-%m-%d_%H %M-%S_%f")}'
                dir_name = '../results/train_and_run/' + dir_name
                os.makedirs(dir_name, exist_ok=True)
                abs_dir_path = os.path.abspath(dir_name)
                return abs_dir_path

        temp_dir = make_calc_dir()

        # copy exec file to the folder for help:
        import shutil
        shutil.copy('train_and_run.py', temp_dir + '/train_and_run.py')

        os.chdir(temp_dir)
        print("Saving data to directory: ", temp_dir)



        import sys
        log_file = temp_dir + '/LOG_GPU.log'
        # sys.stdout = open(log_file, 'w')

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        data_folder = "/home/qklmn/data/"
        # data_folder = "/data1/simulations/"

        print("Loading training data...")
        # traj_300 = io.read("/data1/simulations/datasets/rotors/different_temperatures/300/OUTCAR", format="vasp-out", index = ":")
        # traj_600 = io.read("/data1/simulations/datasets/rotors/different_temperatures/600/OUTCAR", format="vasp-out", index = ":")
        # traj_900 = io.read("/data1/simulations/datasets/rotors/different_temperatures/900/OUTCAR", format="vasp-out", index = ":")
        # traj_1200 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
        # traj_1500 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
        # traj_1800 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1800/OUTCAR", format="vasp-out", index = ":")
        # traj_2100 = io.read("/data1/simulations/datasets/rotors/different_temperatures/2100/OUTCAR", format="vasp-out", index = ":")
        # print(len(traj_300), len(traj_600), len(traj_900), len(traj_1200), len(traj_1500), len(traj_1800), len(traj_2100))


        traj_300 = io.read(data_folder + "datasets/rotors/different_temperatures/300/OUTCAR", format="vasp-out", index = ":")
        traj_600 = io.read(data_folder + "datasets/rotors/different_temperatures/600/OUTCAR", format="vasp-out", index = ":")
        # traj_900 = io.read(data_folder + "datasets/rotors/different_temperatures/900/OUTCAR", format="vasp-out", index = ":")
        # traj_1200 = io.read(data_folder + "datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
        traj_1500 = io.read(data_folder + "datasets/rotors/different_temperatures/1500/OUTCAR", format="vasp-out", index = ":")
        traj_1800 = io.read(data_folder + "datasets/rotors/different_temperatures/1800/OUTCAR", format="vasp-out", index = ":")
        traj_2100 = io.read(data_folder + "datasets/rotors/different_temperatures/2100/OUTCAR", format="vasp-out", index = ":")

        # for parameter selection purpose:
        # traj_train = traj_300[100:500:5].copy() #traj_1800.copy() + traj_2100.copy()
        traj_train = traj_2100[100:500:20].copy()
        # traj_train = traj_2100.copy()
        # training_indices = np.sort(  np.arange(0, 500, 5) )  
        # traj_train = [traj_md[i] for i in training_indices]

        traj_test = traj_300[400:420].copy()
        # test_indices = np.sort(  np.random.choice(np.arange(0,92795), 200, replace=False) ) 
        # test_indices = np.sort(  np.arange(400,410,1) ) 
        # traj_test = [traj_md[i] for i in test_indices]



        from fande.data import FandeDataModuleASE

        ## Train data:
        energies_train = np.zeros(len(traj_train) )
        forces_train = np.zeros( (len(traj_train), len(traj_train[0]), 3 ) )
        for i, snap in enumerate(traj_train):
                energies_train[i] = snap.get_potential_energy()
                forces_train[i] = snap.get_forces()
        train_data = {'trajectory': traj_train, 'energies': energies_train, 'forces': forces_train}
        ## Test data:
        energies_test = np.zeros(len(traj_test) )
        forces_test = np.zeros( (len(traj_test), len(traj_test[0]), 3 ) )
        for i, snap in enumerate(traj_test):
                energies_test[i] = snap.get_potential_energy()
                forces_test[i] = snap.get_forces()
        test_data = {'trajectory': traj_test, 'energies': energies_test, 'forces': forces_test}


        atomic_groups = [ind_H_1, ind_H_2, ind_H_3, ind_H_4, ind_C_1, ind_C_2, ind_C_3, ind_C_4, ind_C_5, ind_C_6, ind_C_7, ind_N_1, ind_O_1, ind_Si_1] 

        train_centers_positions = sum(atomic_groups, []) #list(range(len(atoms)))
        train_derivatives_positions = sum(atomic_groups, [])#list(range(len(atoms)))


        fdm = FandeDataModuleASE(train_data, test_data, hparams)

        fdm.calculate_invariants_librascal(
        soap_params,
        atomic_groups = atomic_groups,
        centers_positions = train_centers_positions, 
        derivatives_positions = train_derivatives_positions,
        same_centers_derivatives=True,
        frames_per_batch=1,
        calculation_context="train")

        fdm.calculate_invariants_librascal(
        soap_params,
        atomic_groups = atomic_groups,
        centers_positions = train_centers_positions, 
        derivatives_positions = train_derivatives_positions,
        same_centers_derivatives=True,
        frames_per_batch=1,
        calculation_context="test")



        for g in range(len(atomic_groups)):
                print("\n-----------")
                print("Group ", g)
                print("-----------")
                plt.plot(fdm.train_F[g].cpu()[1::1], linestyle = 'None', marker='o', label='train')
                plt.plot(fdm.test_F[g].cpu()[1::1], linestyle = 'None', marker='x', label='test')
                plt.savefig("tran_test_forces_group_" + str(g) + ".png")
                plt.close()

                plt.hist(fdm.test_F[g].cpu().numpy())
                plt.savefig("histogram_forces_group_" + str(g) + ".png")
                plt.close()
                print(f"Number of training points for group {g}: ", fdm.train_DX[g].shape[-2])
                print(f"Number of test points for group {g}: ", fdm.test_DX[g].shape[-2])



        # import torch
        # import numpy as np

        # total_training_random_samples = 10
        # high_force_samples = 5
        # random_samples = total_training_random_samples - high_force_samples

        # indices_high_force = torch.concat( 
        #     (torch.topk(fdm.test_F[0], high_force_samples//2, largest=True)[1],  
        #      torch.topk(fdm.test_F[0], high_force_samples//2, largest=False)[1]) ).cpu().numpy()

        # ind_slice = np.sort(  
        #     np.random.choice(np.setdiff1d(np.arange(0, fdm.train_F[0].shape[0]), indices_high_force), random_samples, replace=False) )

        # indices = np.concatenate((ind_slice, indices_high_force))
        # indices = np.unique(indices)
        # print("High force indices: ", indices_high_force)
        # print(ind_slice)


        return fdm


def sample_data(fdm, N_samples):

        import numpy as np
        # seed_everything(42, workers=True)
        import torch


        # N_samples = 10_000
        # N_factor = 1.0

        ### Prepare data loaders and specify how to sample data for each group:
        total_samples_per_group = [
        N_samples, # ind_H_1
        N_samples, # ind_H_2
        N_samples, # ind_H_3
        N_samples, # ind_H_4    
        N_samples, # ind_C_1
        N_samples, # ind_C_2
        N_samples, # ind_C_3
        N_samples, # ind_C_4
        N_samples, # ind_C_5
        N_samples, # ind_C_6
        N_samples, # ind_C_7
        N_samples, # ind_N_1
        N_samples, # ind_O_1
        N_samples, # ind_Si_1 
        ]

        high_force_samples_per_group = [
        0, # ind_H_1
        0, # ind_H_2
        0, # ind_H_3
        0, # ind_H_4    
        0, # ind_C_1
        0, # ind_C_2
        0, # ind_C_3
        0, # ind_C_4
        0, # ind_C_5
        0, # ind_C_6
        0, # ind_C_7
        0, # ind_N_1
        0, # ind_O_1
        0, # ind_Si_1
        ]

        train_data_loaders = fdm.prepare_train_data_loaders(
        total_samples_per_group=total_samples_per_group,
        high_force_samples_per_group=high_force_samples_per_group)

        # hparams['train_indices'] = fdm.train_indices


        return train_data_loaders



def prepare_model(train_data_loaders, hparams, soap_params, n_steps, learning_rate):

        lr = learning_rate

        import logging

        logging.getLogger("pytorch_lightning").setLevel(logging.INFO) # logging.ERROR to disable or INFO

        model_H_1_hparams = {
        'atomic_group' : ind_H_1,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[0].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_H_2_hparams = {
        'atomic_group' : ind_H_2,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[1].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_H_3_hparams = {
        'atomic_group' : ind_H_3,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[2].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_H_4_hparams = {
        'atomic_group' : ind_H_4,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[3].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        hparams_models_H = [model_H_1_hparams, model_H_2_hparams, model_H_3_hparams, model_H_4_hparams]

        model_C_1_hparams = {
        'atomic_group' : ind_C_1,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[4].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_2_hparams = {
        'atomic_group' : ind_C_2,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[5].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_3_hparams = {
        'atomic_group' : ind_C_3,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[6].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_4_hparams = {
        'atomic_group' : ind_C_4,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[7].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_5_hparams = {
        'atomic_group' : ind_C_5,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[8].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_6_hparams = {
        'atomic_group' : ind_C_6,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[9].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        model_C_7_hparams = {
        'atomic_group' : ind_C_7,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[10].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }


        hparams_models_C =  [model_C_1_hparams, model_C_2_hparams, model_C_3_hparams, model_C_4_hparams, model_C_5_hparams,  model_C_6_hparams,  model_C_7_hparams]

        model_N_1_hparams = {
        'atomic_group' : ind_N_1,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[11].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        hparams_models_N =  [model_N_1_hparams]

        model_O_1_hparams = {
        'atomic_group' : ind_O_1,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[12].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        hparams_models_O =  [model_O_1_hparams]


        model_Si_1_hparams = {
        'atomic_group' : ind_Si_1,
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps,
        'learning_rate' : lr,
        'soap_dim' : train_data_loaders[13].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }

        hparams_models_Si =  [model_Si_1_hparams]

        # model_Si_hparams = {
        #     'atomic_group' : Si_atoms,
        #     'dtype' : hparams['dtype'],
        #     'device' : hparams['device'],
        #     'num_epochs' : 1, 
        #     'learning_rate' : 0.01,
        #     'soap_dim' : fdm.train_DX[1].shape[-1],
        #     'soap_params' : soap_params,
        # }


        hparams['per_model_hparams'] = [ 
        model_H_1_hparams,
        model_H_2_hparams,
        model_H_3_hparams,
        model_H_4_hparams, 
        model_C_1_hparams,
        model_C_2_hparams,
        model_C_3_hparams,
        model_C_4_hparams,
        model_C_5_hparams,
        model_C_6_hparams,
        model_C_7_hparams,
        model_N_1_hparams,
        model_O_1_hparams,
        model_Si_1_hparams
        ] # access per_model_hparams by model.model_id

        # hparams['soap_dim'] = fdm.train_DX[0].shape[-1]


        #####################################################################

        from fande.models import ModelForces, GroupModelForces, ModelEnergies, MyCallbacks

        model_H_1 = ModelForces(
        train_x = train_data_loaders[0].dataset[:][0],
        train_y = train_data_loaders[0].dataset[:][1],
        atomic_group = ind_H_1,
        hparams = hparams,
        id=0)

        model_H_2 = ModelForces(
        train_x = train_data_loaders[1].dataset[:][0],
        train_y = train_data_loaders[1].dataset[:][1],
        atomic_group = ind_H_2,
        hparams = hparams,
        id=1)

        model_H_3 = ModelForces(
        train_x = train_data_loaders[2].dataset[:][0],
        train_y = train_data_loaders[2].dataset[:][1],
        atomic_group = ind_H_3,
        hparams = hparams,
        id=2)

        model_H_4 = ModelForces(
        train_x = train_data_loaders[3].dataset[:][0],
        train_y = train_data_loaders[3].dataset[:][1],
        atomic_group = ind_H_4,
        hparams = hparams,
        id=3)



        model_C_1 = ModelForces(
        train_x = train_data_loaders[4].dataset[:][0],
        train_y = train_data_loaders[4].dataset[:][1],
        atomic_group = ind_C_1,
        hparams = hparams,
        id=4)

        model_C_2 = ModelForces(
        train_x = train_data_loaders[5].dataset[:][0],
        train_y = train_data_loaders[5].dataset[:][1],
        atomic_group = ind_C_2,
        hparams = hparams,
        id=5)

        model_C_3 = ModelForces(
        train_x = train_data_loaders[6].dataset[:][0],
        train_y = train_data_loaders[6].dataset[:][1],
        atomic_group = ind_C_3,
        hparams = hparams,
        id=6)

        model_C_4 = ModelForces(
        train_x = train_data_loaders[7].dataset[:][0],
        train_y = train_data_loaders[7].dataset[:][1],
        atomic_group = ind_C_4,
        hparams = hparams,
        id=7)

        model_C_5 = ModelForces(
        train_x = train_data_loaders[8].dataset[:][0],
        train_y = train_data_loaders[8].dataset[:][1],
        atomic_group = ind_C_5,
        hparams = hparams,
        id=8)

        model_C_6 = ModelForces(
        train_x = train_data_loaders[9].dataset[:][0],
        train_y = train_data_loaders[9].dataset[:][1],
        atomic_group = ind_C_6,
        hparams = hparams,
        id=9)

        model_C_7 = ModelForces(
        train_x = train_data_loaders[10].dataset[:][0],
        train_y = train_data_loaders[10].dataset[:][1],
        atomic_group = ind_C_7,
        hparams = hparams,
        id=10)



        model_N_1 = ModelForces(
        train_x = train_data_loaders[11].dataset[:][0],
        train_y = train_data_loaders[11].dataset[:][1],
        atomic_group = ind_N_1,
        hparams = hparams,
        id=11)


        model_O_1 = ModelForces(
        train_x = train_data_loaders[12].dataset[:][0],
        train_y = train_data_loaders[12].dataset[:][1],
        atomic_group = ind_O_1,
        hparams = hparams,
        id=12)

        model_Si_1 = ModelForces(
        train_x = train_data_loaders[13].dataset[:][0],
        train_y = train_data_loaders[13].dataset[:][1],
        atomic_group = ind_Si_1,
        hparams = hparams,
        id=13)




        AG_force_model = GroupModelForces(
        models= [
                model_H_1, 
                model_H_2, 
                model_H_3, 
                model_H_4, 
                model_C_1, 
                model_C_2, 
                model_C_3, 
                model_C_4, 
                model_C_5,
                model_C_6, 
                model_C_7, 
                model_N_1,
                model_O_1, 
                model_Si_1,
                ], # model_N, model_O, model_Si],
        train_data_loaders = train_data_loaders,
        hparams=hparams)

        AG_force_model.fit()


        return AG_force_model




def prepare_fande_ase_calc(hparams, soap_params):

        from fande.predict import PredictorASE
        from fande.ase import FandeCalc


        fdm = prepare_data(hparams, soap_params)

        train_data_loaders = sample_data(fdm, N_samples=1000)

        AG_force_model = prepare_model(train_data_loaders, hparams, soap_params, 2, 0.1)

        predictor = PredictorASE(
                    fdm,
                    AG_force_model,
                    hparams,
                    soap_params
        )


        fande_calc = FandeCalc(predictor)


        return fande_calc