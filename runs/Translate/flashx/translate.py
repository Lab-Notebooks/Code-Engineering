# Prompt engineering for building diffusion stencils for constant and variable coefficient equation

# Import libraries
from typing import Optional
import fire
from llama import Llama
import yaml
import os


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 4096,
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
        f'Welcome to Llama 2 chat interface for code translation for Flash-X. The source code for Flash-X is located at {os.getenv("FLASHX_HOME")}'
    )
    print("\n")

    print("Model Configuration:")
    print(f"ckpt_path: {ckpt_dir}")
    print(f"tokenizer_path: {tokenizer_path}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"max_batch_size: {max_batch_size}")
    print("\n")

    instructions = [dict(role="system", content="Translate FORTRAN code to C++")]

    while True:
        filename = input(f"Flash-X source file (or exit): ")

        if filename.upper() == "EXIT":
            break

        with open(os.getenv("FLASHX_HOME") + os.sep + filename, "r") as source:
            prompt = "".join(source.readlines())

        print(prompt)

        instructions.append(dict(role="user", content=prompt))

        results = generator.chat_completion(
            [instructions],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for result in results:
            print(
                f"{result['generation']['role'].upper()}: {result['generation']['content']}"
            )
            print("")

            savefile = filename.replace(".F90", ".cpp")

            with open(os.getenv("FLASHX_HOME") + os.sep + savefile, "w") as dest:
                dest.write(result["generation"]["content"])

        instructions.pop(-1)


if __name__ == "__main__":
    fire.Fire(main)
