# Bash script for `jobrunner` to install AMReX

# Setup AMReX
if [ ! -d "Llama-Chat" ]; then
	git clone git@github.com:randaller/llama-chat.git --branch main Llama-Chat
fi

cd Llama-Chat && pip3 install -r requirements.txt && pip3 install -e .
