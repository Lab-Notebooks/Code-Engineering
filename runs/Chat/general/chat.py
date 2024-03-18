# Prompt engineering for building diffusion stencils for constant and variable coefficient equation

# Import libraries
from typing import Optional
import fire
from llama import Llama
import json

from types import SimpleNamespace

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


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print(
        f"{Color.red}Welcome to a simple chat interface for Llama2 model for software engineering applications{Color.end}"
    )
    print("\n")

    chat_name = input(f"{Color.red}Transcript-Name:{Color.end} ")
    print("\n")

    print(f"{Color.red}temperature:{Color.end} {temperature}")
    print(f"{Color.red}top_p:{Color.end} {top_p}")
    print(f"{Color.red}max_seq_len:{Color.end} {max_seq_len}")
    print(f"{Color.red}max_gen_len:{Color.end} {max_gen_len}")
    print(f"{Color.red}max_batch_size:{Color.end} {max_batch_size}")
    print("\n")

    instructions = [
        dict(role="system", content="Provide answers in code when appropriate")
    ]

    while True:
        prompt = input(f"{Color.red}USER:{Color.end} ")

        if prompt.upper() == "EXIT":
            break

        instructions.append(dict(role="user", content=prompt))

        results = generator.chat_completion(
            [instructions],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for result in results:
            print(
                f"{Color.red}{result['generation']['role'].upper()}:{Color.end} {result['generation']['content']}"
            )
            print("")
            instructions.append(result["generation"])

    with open(f"{chat_name}.json", "w") as outfile:
        json.dump(
            {
                "configuration": dict(
                    ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    temperature=temperature,
                    top_p=top_p,
                    max_seq_len=max_seq_len,
                    max_gen_len=max_gen_len,
                    max_batch_size=max_batch_size,
                ),
                "chat": instructions,
            },
            outfile,
            indent=2,
        )


if __name__ == "__main__":
    fire.Fire(main)
