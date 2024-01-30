from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {
                "role": "system",
                "content": "Provide answers in FORTRAN",
            },
            {
                "role": "user",
                "content": "Write a subroutine to numerically compute diffusion term in x-y direction for a variable phi on a cell-centered grid. Assume constant coefficients and store the results in variable rhs which should be an input to the subroutine.",
            }
        ],
        [
            {
                "role": "system",
                "content": "Provide answers in FORTRAN",
            },
            {
                "role": "user",
                "content": "Write a subroutine to numerically compute variable coefficient diffusion term in x-y directions for a variable phi on a staggered finite difference mesh. Use coef as a variable to store those coefficients. Assume that phi is located on cell-centers, coef located on face-centers, and the result rhs located on cell-centers.",
            }
        ],
 
    ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
