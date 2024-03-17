# Prompt engineering for building diffusion stencils for constant and variable coefficient equation

# Import libraries
from typing import Optional
import fire
from llama import Llama
import json


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
        "Welcome to a simple chat interface for Llama2 model for software engineering applications"
    )
    print("\n")

    chat_name = input(f"Chat Transcript Name: ")
    print("\n")

    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"max_gen_len: {max_gen_len}")
    print(f"max_batch_size: {max_batch_size}")
    print("\n")

    instructions = [
        dict(role="system", content="Provide answers in code when appropriate")
    ]

    while True:
        prompt = input(f"USER: ")

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
                f"{result['generation']['role'].upper()}: {result['generation']['content']}"
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
