# CS6540-Final-Project

## How-To for Queueing VACC jobs

See 3_cropped/process_videos.sh for an example - you may need to change file paths, etc, but the important bit is the #!/bin/bash at the start and the #SBATCH commands.

These specify the resources for your job, partition to run on, output and error logs, and job name (useful for canceling). After the #SBATCH commands, you are in CLI and can call anything you'd like. In process_videos.sh there is a call to python within miniconda3, which you may need to change for your purposes/name of the environment. To note, change the partition to "dggpu" for deepgreen, "bdgpu" for blackdiamond, and "bluemoon" for bluemoon partition. See knowledge base for more info.

To run the job, call "sbatch filename". To check on the job, call "squeue -u netid". You can check on a partition using "squeue -p partition_name". 

To cancel a job, call "scancel -n job_name" to cancel all jobs for a name. If you want to cancel all your jobs, call "scancel -u netid".

Again, the VACC knowledge base is a good place to go for any extra information.
