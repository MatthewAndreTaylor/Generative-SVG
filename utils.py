import torch
from svgpathtools import svgstr2paths


def svg_to_tensor(svg_content: str, max_sequence_length: int = 200):
    """
    Convert SVG path data to a tensor of (x, y, flag),
    where flag=0 for a move (start of path), 1 for a line continuation.
    """
    paths, _ = svgstr2paths(svg_content)
    tensor = torch.zeros((max_sequence_length, 3))  # (seq_len, 3)
    idx = 0

    for path in paths:
        if idx >= max_sequence_length:
            break

        # Start point (move)
        tensor[idx, 0] = path[0].start.real
        tensor[idx, 1] = path[0].start.imag
        tensor[idx, 2] = 0.0
        idx += 1

        # Segments
        for seg in path:
            if idx >= max_sequence_length:
                break
            tensor[idx, 0] = seg.end.real
            tensor[idx, 1] = seg.end.imag
            tensor[idx, 2] = 1.0
            idx += 1

    return tensor


def tensor_to_svg(tensor: torch.Tensor, size=256, stroke_width=0.8) -> str:
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}"><g stroke-width="{stroke_width}">'
    ]

    path_cmds = []

    for i in range(tensor.shape[0]):
        x, y, flag = tensor[i].tolist()

        # Skip unused rows (all zeros)
        if x == 0 and y == 0 and flag == 0:
            continue

        if flag == 0.0:
            # Start new path
            if path_cmds:
                path_str = " ".join(path_cmds)
                svg_parts.append(
                    f'<path d="{path_str}" stroke="black" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
                )
                path_cmds = []

            path_cmds.append(f"M {x} {y}")
        else:
            path_cmds.append(f"L {x} {y}")

    # Flush last path
    if path_cmds:
        path_str = " ".join(path_cmds)
        svg_parts.append(
            f'<path d="{path_str}" stroke="black" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg_parts.append("</g></svg>")
    return "\n".join(svg_parts)
