# Bash script for `jobrunner` to install AMReX

# Setup AMReX
if [ ! -d "Flash-X" ]; then
	git clone git@github.com:Flash-X/Flash-X --branch main Flash-X
fi
