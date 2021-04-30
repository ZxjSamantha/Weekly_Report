1. custome imports: 
```
from htnet_model import htnet
from model_utils import load_data, folds_choose_subjects, subject_data_inds, roi_proj_rf, str2bool, get_custom_motor_rois, proj_mats_good_rois
```

---

2. run_nn_models:

parameters:

1. sp: save path

2. n_folds: number of folds (per participant)

3. combined_sbs, 

4. lp: load path 

5. roi_proj_loadpath: projection matrix load path

6. pats_ids_in: ID of participants

7. n_evs_per_sbj: ?

8. test_day:

9. tlim = `[-1,1]`

10. n_chan_all: ?

11. dipole_dens_thresh: ?

12. rem_bad_chans: ?

13. models: 

14. save_suffix: 

