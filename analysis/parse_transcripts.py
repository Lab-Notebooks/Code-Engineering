import fire
import json
from types import SimpleNamespace
import os

Color = SimpleNamespace(
    purple="\033[95m",
    cyan="\033[96m",
    darkcyan="\033[36m",
    blue="\033[94m",
    green="\033[92m",
    yellow="\033[93m",
    red="\033[91m",
    bold="\033[1m",
    underline="\033[4m",
    end="\033[0m",
)


def main(json_file: str ):

    with open(json_file, "r") as inputfile:
        json_dict = json.load(inputfile)
        configuration = json_dict["configuration"]
        chat = json_dict["chat"]

    for entry in chat:
        if entry["role"].lower() == "user":
            print(f'{Color.red}USER: {entry["content"]}{Color.end}')
        elif entry["role"].lower() == "assistant":
            print(f'{Color.blue}ASSITANT: {entry["content"]}{Color.end}')
            print("")
        elif entry["role"].lower() == "system":
            print(f'{Color.darkcyan}SYSTEM: {entry["content"]}{Color.end}')

if __name__ == "__main__":
    fire.Fire(main)
