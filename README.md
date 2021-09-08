# lcquickfit: rapid transit fitting with eccentricity constraints
Note: Currently only works for TOI targets - soon to be generalized

## Installing dependencies
Create new environment and install dependencies by running the following at the top level of this repo:
- *\$ conda env create -f environment.yml*
- *\$ conda activate lcquickfit_env*

## To run code
'duration' mode: fit for eccentricity indirectly and more rapidly by sampling in duration and deriving eccentricity constraints post-sampling
- *\$ ~/opt/anaconda3/envs/lcquickfit_new_env/bin/python3 lcquickfit.py \<TOI number\> \</your/preferred/output/path/\> duration*

Example: 
- *\$ ~/opt/anaconda3/envs/lcquickfit_new_env/bin/python3 lcquickfit.py 1272 /Users/mason/lcquickfit/ duration*
- Takes ~30 mins to run
  
'full' mode: fit for eccentricity directly by sampling in sqrt(e)cos(omega) & sqrt(e)sin(omega)
- *\$ ~/opt/anaconda3/envs/lcquickfit_new_env/bin/python3 lcquickfit.py \<TOI number\> \</your/preferred/output/path/\> full*

Example: 
- *\$ ~/opt/anaconda3/envs/lcquickfit_new_env/bin/python3 lcquickfit.py 1272 /Users/mason/lcquickfit/ full*
- Takes ~2-3 hours to run
  
## To run code in segments
- TBA
