# Prompt engineering for building diffusion stencils for constant and variable coefficient equation

# Import libraries
from typing import Optional
import fire
from llama import Llama


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

    instructions = [dict(role="system", content="Provide answers in code when appropriate")]

    while True:
        prompt = input(f"USER: ")
        instructions.append(dict(role="user", content=prompt))

        results = generator.chat_completion(
            [instructions], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
        )

        for result in results:
            print(f"{result['generation']['role'].upper()}: {result['generation']['content']}")
            print("\n")
            instructions.append(result['generation'])


if __name__ == "__main__":
    fire.Fire(main)
