# research-reproduction
A personal lab for replicating and experimenting with cutting-edge research in Machine Learning, Federated Learning, Differential Privacy, and GAN-based synthetic data generation.

Complete this form if you require an Iridis (high performance computing) account

https://sotonproduction.service-now.com/serviceportal?id=sc_cat_item&sys_id=bce3a6fa1bf34210e3076351f54bcbe9

The guidebook of HPC

https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Submitting-Jobs-Slurm.aspx

# How to use Iridis HPC

scp -r /Users/shuyanrocky/Downloads/test sz1c24@loginX001.iridis.soton.ac.uk:/home/sz1c24/ # Upload folder to remote HPC

ssh sz1c24@loginX001.iridis.soton.ac.uk # Login

within terminal:

chmod +x hpc528.sh # Grant execution permissions before run .sh file

sbatch hpc528.sh # Submit a job script to the scheduler

squeue -lu sz1c24  # Show all jobs

scontrol show job 123456 # show the job 123456
