# Bash script for `jobrunner` to install AMReX

# Setup AMReX
if [ ! -d "CodeLlama" ]; then
	git clone git@github.com:facebookresearch/codellama.git --branch main CodeLlama && cd CodeLlama
else
	cd CodeLlama
fi

./download.sh
