# Bash script for `jobrunner` to install AMReX

# Setup AMReX
if [ ! -d "Llama" ]; then
	git clone git@github.com:meta-llama/llama.git --branch main Llama
fi

cd Llama && pip3 install -e .
