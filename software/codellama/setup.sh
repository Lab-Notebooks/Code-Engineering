# Bash script for `jobrunner` to install AMReX

# Setup AMReX
if [ ! -d "Codellama" ]; then
	git clone git@github.com:facebookresearch/codellama.git --branch main Codellama && cd Codellama
else
	cd Codellama
fi
