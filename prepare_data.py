# Matthew Taylor 2025

import re
from svgpathtools import (
    svgstr2paths,
    paths2Drawing,
    Path,
    CubicBezier,
    Line,
    QuadraticBezier,
    Arc,
)

# Utilities for optimizing SVG files
_viewbox_re = re.compile(r'viewBox\s*=\s*"([^"]+)"')


def parse_viewbox(svg_content: str):
    m = _viewbox_re.search(svg_content)
    if not m:
        raise ValueError("SVG file does not contain a viewBox attribute.")
    parts = list(map(float, m.group(1).split()))
    if len(parts) != 4:
        raise ValueError(f"Invalid viewBox format: {m.group(1)}")
    min_x, min_y, width, height = parts
    return min_x, min_x + width, min_y, min_y + height


def make_quantizer(min_x, max_x, min_y, max_y, bins):
    if max_x > min_x:
        scale_x = (bins - 1) / (max_x - min_x)
    else:
        scale_x = 0.0

    if max_y > min_y:
        scale_y = (bins - 1) / (max_y - min_y)
    else:
        scale_y = 0.0

    def quantize_point(p):
        qx = (p.real - min_x) * scale_x
        qy = (p.imag - min_y) * scale_y
        qx = max(0, min(int(round(qx)), bins - 1))
        qy = max(0, min(int(round(qy)), bins - 1))
        return complex(qx, qy)

    return quantize_point


def quantize_paths(paths, bins, svg_content):
    min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
    quantize_point = make_quantizer(min_x, max_x, min_y, max_y, bins)
    # print(f"Quantizing paths with viewbox: {min_x}, {max_x}, {min_y}, {max_y} and bins: {bins}")

    quantized_paths = []
    for path in paths:
        qpath = Path()
        for segment in path:
            if isinstance(segment, Line):
                q_start = quantize_point(segment.start)
                q_end = quantize_point(segment.end)
                qpath.append(Line(q_start, q_end))

            elif isinstance(segment, QuadraticBezier):
                q_start = quantize_point(segment.start)
                q_ctrl = quantize_point(segment.control)
                q_end = quantize_point(segment.end)
                qpath.append(QuadraticBezier(q_start, q_ctrl, q_end))

            elif isinstance(segment, CubicBezier):
                q_start = quantize_point(segment.start)
                q_ctrl1 = quantize_point(segment.control1)
                q_ctrl2 = quantize_point(segment.control2)
                q_end = quantize_point(segment.end)
                qpath.append(CubicBezier(q_start, q_ctrl1, q_ctrl2, q_end))

            elif isinstance(segment, Arc):
                # Convert arc to cubic and quantize
                for cubic in segment.as_cubic_curves():
                    q_start = quantize_point(cubic.start)
                    q_ctrl1 = quantize_point(cubic.control1)
                    q_ctrl2 = quantize_point(cubic.control2)
                    q_end = quantize_point(cubic.end)
                    qpath.append(CubicBezier(q_start, q_ctrl1, q_ctrl2, q_end))

            elif isinstance(segment, Arc):
                # Convert arc to cubic and quantize
                for cubic in segment.as_cubic_curves():
                    q_start = quantize_point(cubic.start)
                    q_ctrl1 = quantize_point(cubic.control1)
                    q_ctrl2 = quantize_point(cubic.control2)
                    q_end = quantize_point(cubic.end)
                    qpath.append(CubicBezier(q_start, q_ctrl1, q_ctrl2, q_end))

            else:
                raise TypeError(f"Unsupported segment type: {type(segment)}")

        quantized_paths.append(qpath)
    return quantized_paths


def clean_svg(svg_content, bins):
    # Clean up XML declaration
    svg_text = re.sub(r"<\?xml[^>]+\?>", "", svg_content).strip()

    # Remove unused xmlns declarations
    svg_text = re.sub(r'\s+xmlns:[^=]+="[^"]+"', "", svg_text)

    # Remove px units from width/height
    # svg_text = re.sub(r'(\bwidth|height)="([^"]+?)px"', r'\1="\2"', svg_text)

    # Remove `.0` from floats like 123.0 → 123
    svg_text = re.sub(r"(\d+)\.0(?=[^0-9])", r"\1", svg_text)

    # Remove empty <defs/>
    svg_text = svg_text.replace("<defs/>", "")

    # remove any width and height attributes
    svg_text = re.sub(r'\s+(width|height)="[^"]*"', "", svg_text)

    # remove any version attributes
    svg_text = re.sub(r'\s+version="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+baseProfile="[^"]*"', "", svg_text)

    # set the viewBox attribute back to the number of bins
    # min_x, max_x, min_y, max_y = 0, bins, 0, bins
    # min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y)
    # viewbox_value = f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}"
    # svg_text = re.sub(r'viewBox="[^"]*"', f'viewBox="{viewbox_value}"', svg_text)

    # should follow the pattern d="[Letter][NUMBER][,][NUMBER][SPACE][Letter|Number][,][NUMBER]..."
    # note there is always a space after each set of numbers except the last one
    # remove all empty spaces after each letter in each path
    svg_text = re.sub(r"([MmLlCcSsQqTtAaZz])\s+", r"\1", svg_text)

    # remove all empty spaces before each single letter in each path
    svg_text = re.sub(r"\s+([MmLlCcSsQqTtAaZz])(?![A-Za-z])", r"\1", svg_text)

    ### OPTIONAL
    # remove fill and stroke attributes
    # svg_text = re.sub(r'\s+fill="[^"]*"', "", svg_text)
    # svg_text = re.sub(r'\s+stroke="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+stroke-width="[^"]*"', "", svg_text)

    # Add a <g> element after <svg ... >
    # TODO: this should be dynamic based on the original stroke width * some factor
    svg_text = re.sub(r"(<svg[^>]*>)", r'\1<g stroke-width="0.4">', svg_text)
    svg_text = svg_text.replace("</svg>", "</g></svg>")

    # remove all whitespace between elements
    svg_text = re.sub(r">\s+<", "><", svg_text)
    return svg_text


def convert_and_quantize_svg(svg_content, bins: int = 128):
    paths, _ = svgstr2paths(svg_content)
    quantized_paths = quantize_paths(paths, bins, svg_content)

    # Use paths2Drawing to get Drawing object, then write to string
    dwg = paths2Drawing(quantized_paths)
    svg_content = dwg.tostring()
    output = clean_svg(svg_content, bins=bins)
    return output


# Utilities for converting QuickDraw sketches to SVG


def quickdraw_to_svg(drawing, stroke_width=0.6, size=256):
    svg_parts = [f'<svg viewBox="0 0 {size} {size}"><g stroke-width="{stroke_width}">']
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        if not xs or not ys:
            continue

        path_cmds = [f"M {xs[0]} {ys[0]}"]
        for x, y in zip(xs[1:], ys[1:]):
            path_cmds.append(f"L {x} {y}")

        path_str = " ".join(path_cmds)
        svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

    svg_parts.append("</g></svg>")
    return "\n".join(svg_parts)
