To generate the required pickle files, update the locations of files in the consts.py file and then run `forced_alignment.py`. 

Ensure [Gentle Aligner](https://github.com/lowerquality/gentle) is installed and the `install.sh` has been run for your virtual environment. More information is in the top level README. 

Add the flag `--mmfusion-legacy` to generate the labels in a format compatible with the mmfusion model. For this project, do not run this flag.
