
# Fish auto encoders

# our regular auto encoders which tend to be less regular
sbatch trainfish_with_args.slrm "" 0.0 0
sbatch trainfish_with_args.slrm "" 0.1 0
sbatch trainfish_with_args.slrm "" 0.5 0
sbatch trainfish_with_args.slrm "" 0.9 0

# our variational auto encoders which tend to be more regular
sbatch trainfish_with_args.slrm "v" 0.0 0
sbatch trainfish_with_args.slrm "v" 0.1 0
sbatch trainfish_with_args.slrm "v" 0.5 0
sbatch trainfish_with_args.slrm "v" 0.9 0

# wacky hijinx with alternating z and x loss function
sbatch trainfish_with_args.slrm "" 1.0 5
sbatch trainfish_with_args.slrm "v" 1.0 5
