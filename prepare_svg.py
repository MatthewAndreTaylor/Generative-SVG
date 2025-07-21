# Matthew Taylor 2025

import os
import re
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths2, wsvg, Path, CubicBezier, Line, QuadraticBezier, Arc

#TODO: some svg's don't work
def parse_viewbox(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox is None:
        raise ValueError("SVG file does not contain a viewBox attribute.")
    parts = list(map(float, viewbox.strip().split()))
    if len(parts) != 4:
        raise ValueError(f"Invalid viewBox format: {viewbox}")
    min_x, min_y, width, height = parts
    max_x = min_x + width
    max_y = min_y + height
    return min_x, max_x, min_y, max_y


def quantize_point(p, min_x, max_x, min_y, max_y, bins):
    x = (p.real - min_x) / (max_x - min_x) if max_x > min_x else 0
    y = (p.imag - min_y) / (max_y - min_y) if max_y > min_y else 0
    qx = min(int(round(x * (bins - 1))), bins - 1)
    qy = min(int(round(y * (bins - 1))), bins - 1)
    return complex(qx, qy)


def quantize_paths(paths, bins, input_file):
    min_x, max_x, min_y, max_y = parse_viewbox(input_file)
    # print(f"Quantizing paths with viewbox: {min_x}, {max_x}, {min_y}, {max_y} and bins: {bins}")

    quantized_paths = []
    for path in paths:
        qpath = Path()
        for segment in path:
            if isinstance(segment, Line):
                q_start = quantize_point(
                    segment.start, min_x, max_x, min_y, max_y, bins
                )
                q_end = quantize_point(segment.end, min_x, max_x, min_y, max_y, bins)
                qpath.append(Line(q_start, q_end))

            elif isinstance(segment, QuadraticBezier):
                q_start = quantize_point(
                    segment.start, min_x, max_x, min_y, max_y, bins
                )
                q_ctrl = quantize_point(
                    segment.control, min_x, max_x, min_y, max_y, bins
                )
                q_end = quantize_point(segment.end, min_x, max_x, min_y, max_y, bins)
                qpath.append(QuadraticBezier(q_start, q_ctrl, q_end))

            elif isinstance(segment, CubicBezier):
                q_start = quantize_point(
                    segment.start, min_x, max_x, min_y, max_y, bins
                )
                q_ctrl1 = quantize_point(
                    segment.control1, min_x, max_x, min_y, max_y, bins
                )
                q_ctrl2 = quantize_point(
                    segment.control2, min_x, max_x, min_y, max_y, bins
                )
                q_end = quantize_point(segment.end, min_x, max_x, min_y, max_y, bins)
                qpath.append(CubicBezier(q_start, q_ctrl1, q_ctrl2, q_end))

            elif isinstance(segment, Arc):
                # Convert arc to cubic and quantize
                for cubic in segment.as_cubic_curves():
                    q_start = quantize_point(
                        cubic.start, min_x, max_x, min_y, max_y, bins
                    )
                    q_ctrl1 = quantize_point(
                        cubic.control1, min_x, max_x, min_y, max_y, bins
                    )
                    q_ctrl2 = quantize_point(
                        cubic.control2, min_x, max_x, min_y, max_y, bins
                    )
                    q_end = quantize_point(cubic.end, min_x, max_x, min_y, max_y, bins)
                    qpath.append(CubicBezier(q_start, q_ctrl1, q_ctrl2, q_end))

            else:
                raise TypeError(f"Unsupported segment type: {type(segment)}")

        quantized_paths.append(qpath)
    return quantized_paths


def clean_svg_file(file_path, bins):
    with open(file_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    # Clean up XML declaration
    svg_text = re.sub(r"<\?xml[^>]+\?>", "", svg_text).strip()

    # Remove unused xmlns declarations
    svg_text = re.sub(r'\s+xmlns:[^=]+="[^"]+"', "", svg_text)

    # Remove px units from width/height
    svg_text = re.sub(r'(\bwidth|height)="([^"]+?)px"', r'\1="\2"', svg_text)

    # Remove `.0` from floats like 123.0 → 123
    svg_text = re.sub(r"(\d+)\.0([^0-9])", r"\1\2", svg_text)
    svg_text = re.sub(r'(\d+)\.0(?=\s|")', r"\1", svg_text)

    # Remove empty <defs/>
    svg_text = svg_text.replace("<defs/>", "")

    # remove any width and height attributes
    svg_text = re.sub(r'\s+(width|height)="[^"]*"', "", svg_text)

    # remove any version attributes
    svg_text = re.sub(r'\s+version="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+baseProfile="[^"]*"', "", svg_text)

    # set the viewBox attribute back to the number of bins
    min_x, max_x, min_y, max_y = 0, bins, 0, bins
    min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y)
    viewbox_value = f"{min_x} {min_y} {max_x - min_x} {max_y - min_y}"
    svg_text = re.sub(r'viewBox="[^"]*"', f'viewBox="{viewbox_value}"', svg_text)

    # should follow the pattern d="[Letter][NUMBER][,][NUMBER][SPACE][Letter|Number][,][NUMBER]..."
    # note there is always a space after each set of numbers except the last one
    # remove all empty spaces after each letter in each path
    svg_text = re.sub(r"([MmLlCcSsQqTtAaZz])\s+", r"\1", svg_text)
    
    # remove all empty spaces before each single letter in each path
    svg_text = re.sub(r"\s+([MmLlCcSsQqTtAaZz])(?![A-Za-z])", r"\1", svg_text)
    
    ### OPTIONAL
    # remove fill and stroke attributes
    svg_text = re.sub(r'\s+fill="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+stroke="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+stroke-width="[^"]*"', "", svg_text)

    # remove all whitespace between elements
    svg_text = re.sub(r">\s+<", "><", svg_text)

    return svg_text


# Main
def convert_and_quantize_svg(input_file, output_file, bins=96):
    paths, _, _ = svg2paths2(input_file)
    quantized_paths = quantize_paths(paths, bins, input_file)
    # num_segments = sum(len(p) for p in quantized_paths)
    # print(f"Total quantized segments: {num_segments}")

    wsvg(quantized_paths, filename=output_file)
    output = clean_svg_file(output_file, bins=bins)
    
    # remove the original file
    os.remove(output_file)
    return output
    
    
    
if __name__ == "__main__":
    input_svg = "apple.svg"
    output_svg = "apple_quantized2.svg"
    convert_and_quantize_svg(input_svg, output_svg)