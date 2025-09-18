import torch
from svgpathtools import svgstr2paths

from prepare_data import parse_viewbox, make_quantizer


# We propose two novel tensor representations for SVG data:

## Stroke-based representation: each row is (dx, dy, flag)
# where dx, dy are the offsets from the previous point,
# and flag is 0 for a "move" (start of new path) and 1 for a "line" (continuation of path).

## Point-based representation: each row is (x, y, flag)
# where x, y are absolute coordinates, and flag is 0 for "move" and 1 for "line".


def svg_strokes_to_tensor_quantized(
    svg_content: str, bins=128, max_sequence_length: int = 200
):
    """
    Convert SVG path data to a quantized *stroke* tensor representation.
    Each row: (dx, dy, flag)
    """
    paths, _ = svgstr2paths(svg_content)
    min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
    quantize_point = make_quantizer(min_x, max_x, min_y, max_y, bins)
    tensor = torch.zeros((max_sequence_length, 3))  # (seq_len, 3)
    max_sequence_length = max_sequence_length - 1  # reserve space for end token
    idx = 0
    prev = None

    for path in paths:
        if idx >= max_sequence_length:
            break

        # Move command (absolute → stored as delta from 0,0 if first move)
        start_quantized = quantize_point(path[0].start)
        dx = start_quantized.real - (prev.real if prev else 0)
        dy = start_quantized.imag - (prev.imag if prev else 0)

        tensor[idx] = torch.tensor([dx, dy, 0.0])
        prev = start_quantized
        idx += 1

        # Line segments
        for seg in path:
            if idx >= max_sequence_length:
                break
            end_quantized = quantize_point(seg.end)
            dx = end_quantized.real - prev.real
            dy = end_quantized.imag - prev.imag
            tensor[idx] = torch.tensor([dx, dy, 1.0])
            prev = end_quantized
            idx += 1
    
    # Mark end of sketch
    tensor[idx, 0] = 0.0
    tensor[idx, 1] = 0.0
    tensor[idx, 2] = -1.0

    return tensor


def svg_strokes_to_tensor(svg_content: str, max_sequence_length: int = 200):
    """
    Convert SVG path data to a *stroke* tensor of (dx, dy, flag).
    flag=0 for move, 1 for line continuation.
    """
    paths, _ = svgstr2paths(svg_content)
    tensor = torch.zeros((max_sequence_length, 3))
    max_sequence_length = max_sequence_length - 1  # reserve space for end token
    idx = 0
    prev = None

    for path in paths:
        if idx >= max_sequence_length:
            break

        # Move
        start = path[0].start
        dx = start.real - (prev.real if prev else 0)
        dy = start.imag - (prev.imag if prev else 0)
        tensor[idx] = torch.tensor([dx, dy, 0.0])
        prev = start
        idx += 1

        # Lines
        for seg in path:
            if idx >= max_sequence_length:
                break
            end = seg.end
            dx = end.real - prev.real
            dy = end.imag - prev.imag
            tensor[idx] = torch.tensor([dx, dy, 1.0])
            prev = end
            idx += 1
            
    
    # Mark end of sketch
    tensor[idx, 0] = 0.0
    tensor[idx, 1] = 0.0
    tensor[idx, 2] = -1.0

    return tensor


def tensor_to_svg_strokes(tensor: torch.Tensor, size=256, stroke_width=0.8) -> str:
    """
    Reconstruct SVG from *stroke* tensor (dx, dy, flag).
    """
    svg_parts = [f'<svg viewBox="0 0 {size} {size}"><g stroke-width="{stroke_width}">']
    path_cmds = []

    x, y = 0.0, 0.0  # start at origin
    for i in range(tensor.shape[0]):
        dx, dy, flag = tensor[i].tolist()
        if flag == -1.0:  # End of sketch
            break

        x += dx
        y += dy

        if flag == 0.0:  # Move
            if path_cmds:
                path_str = " ".join(path_cmds)
                svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')
                path_cmds = []
            path_cmds.append(f"M {x} {y}")
        else:  # Line
            path_cmds.append(f"L {x} {y}")

    # Flush last path
    if path_cmds:
        path_str = " ".join(path_cmds)
        svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

    svg_parts.append("</g></svg>")
    return "\n".join(svg_parts)


def svg_to_tensor_quantized(svg_content: str, bins=128, max_sequence_length: int = 200):
    """
    Convert SVG path data to a quantized tensor representation.
    """
    paths, _ = svgstr2paths(svg_content)
    min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
    quantize_point = make_quantizer(min_x, max_x, min_y, max_y, bins)
    tensor = torch.zeros((max_sequence_length, 3))  # (seq_len, 3)
    max_sequence_length = max_sequence_length - 1  # reserve space for end token
    idx = 0

    for path in paths:
        if idx >= max_sequence_length:
            break

        start_quantized = quantize_point(path[0].start)
        tensor[idx, 0] = start_quantized.real
        tensor[idx, 1] = start_quantized.imag
        # tensor[idx, 2] = 0.0 # no-op, already zero for move
        idx += 1

        for segment in path:
            if idx >= max_sequence_length:
                break

            end_quantized = quantize_point(segment.end)
            tensor[idx, 0] = end_quantized.real
            tensor[idx, 1] = end_quantized.imag
            tensor[idx, 2] = 1.0  # Line command
            idx += 1
            
    
    # Mark end of sketch
    tensor[idx, 0] = 0.0
    tensor[idx, 1] = 0.0
    tensor[idx, 2] = -1.0

    return tensor


def svg_to_tensor(svg_content: str, max_sequence_length: int = 200):
    """
    Convert SVG path data to a tensor of (x, y, flag),
    where flag=0 for a move (start of path), 1 for a line continuation.
    """
    paths, _ = svgstr2paths(svg_content)
    tensor = torch.zeros((max_sequence_length, 3))  # (seq_len, 3)
    max_sequence_length = max_sequence_length - 1  # reserve space for end token
    idx = 0

    for path in paths:
        if idx >= max_sequence_length:
            break

        # Start point (move)
        tensor[idx, 0] = path[0].start.real
        tensor[idx, 1] = path[0].start.imag
        # tensor[idx, 2] = 0.0 # no-op, already zero for move
        idx += 1

        # Segments
        for seg in path:
            if idx >= max_sequence_length:
                break
            tensor[idx, 0] = seg.end.real
            tensor[idx, 1] = seg.end.imag
            tensor[idx, 2] = 1.0
            idx += 1

    # Mark end of sketch
    tensor[idx, 0] = 0.0
    tensor[idx, 1] = 0.0
    tensor[idx, 2] = -1.0

    return tensor


def tensor_to_svg(tensor: torch.Tensor, size=256, stroke_width=0.8) -> str:
    svg_parts = [f'<svg viewBox="0 0 {size} {size}"><g stroke-width="{stroke_width}">']
    path_cmds = []

    for i in range(tensor.shape[0]):
        x, y, flag = tensor[i].tolist()

        # Skip unused rows (all zeros)
        if flag == -1.0:
            break

        if flag == 0.0:
            if path_cmds:
                path_str = " ".join(path_cmds)
                svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')
                path_cmds = []

            path_cmds.append(f"M {x} {y}")
        else:
            path_cmds.append(f"L {x} {y}")

    # Flush last path
    if path_cmds:
        path_str = " ".join(path_cmds)
        svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

    svg_parts.append("</g></svg>")
    return "\n".join(svg_parts)
