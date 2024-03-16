# Prompt engineering for building diffusion stencils for constant and variable coefficient equation

# Import libraries
from typing import Optional
import fire
from llama import Llama


def get_instructions(prompt):
    """
    return instruction list from prompt
    """
    return [
        dict(role="system", content="Provide answers in FORTRAN"),
        dict(role="user", content=prompt),
    ]


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 1024,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    constant_diffusion_prompt = (
        "Write a subroutine to numerically compute diffusion term in x-y "
        + 'direction for a variable "phi" on a cell-centered grid. Assume constant '
        + 'coefficients and store the results in variable "rhs" which should be an '
        + "input to the subroutine."
    )

    # variable_diffusion_prompt = (
    #    "Write a subroutine to numerically compute variable coefficient diffusion "
    #    + 'term in x-y directions for a variable "phi" on a staggered finite difference '
    #    + 'mesh. Use "coeff" as a variable to store those coefficients. Assume that "phi" is '
    #    + 'located on cell-centers, "coeff" located on face-centers, and the result "rhs" '
    #    + "located on cell-centers."
    # )

    variable_diffusion_prompt = (
        "Write a subroutine to numerically compute diffusion "
        + 'term with variable diffusivity in x-y directions for a variable "phi" on a staggered finite difference '
        + 'mesh. Use "coeffx" as a variable for x-face coefficients with spatial bounds (nx+1, ny) and '
        + '"coeffy" as a variable for y-face coefficients with spatial bounds (nx, ny+1). Assume that "phi" is '
        + 'located on cell-centers with bounds (nx, ny), The result "rhs" should also be located on cell-centers '
        + "with bounds (nx, ny). The diffusion term should be consistent with the mathematical expression that computes derivates of variable fluxes."
    )

    instructions = []
    instructions.append(get_instructions(constant_diffusion_prompt))
    instructions.append(get_instructions(variable_diffusion_prompt))

    results = generator.chat_completion(
        instructions, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
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
