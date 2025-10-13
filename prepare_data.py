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

    # remove any width and height attributes
    svg_text = re.sub(r'\s+(width|height)="[^"]*"', "", svg_text)

    # remove any version attributes
    svg_text = re.sub(r'\s+version="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+baseProfile="[^"]*"', "", svg_text)

    # should follow the pattern d="[Letter][NUMBER][,][NUMBER][SPACE][Letter|Number][,][NUMBER]..."
    # note there is always a space after each set of numbers except the last one
    # remove all empty spaces after each letter in each path
    svg_text = re.sub(r"([MmLlCcSsQqAaZz])\s+", r"\1", svg_text)

    # remove all empty spaces before each single letter in each path
    svg_text = re.sub(r"\s+([MmLlCcSsQqAaZz])(?![A-Za-z])", r"\1", svg_text)

    ### OPTIONAL
    # remove fill and stroke attributes
    # svg_text = re.sub(r'\s+fill="[^"]*"', "", svg_text)
    # svg_text = re.sub(r'\s+stroke="[^"]*"', "", svg_text)
    svg_text = re.sub(r'\s+stroke-width="[^"]*"', "", svg_text)

    # Add a <g> element after <svg ... >
    # TODO: this should be dynamic based on the original stroke width * some factor
    svg_text = re.sub(r"(<svg[^>]*>)", r'\1<g stroke-width="1.0">', svg_text)
    svg_text = svg_text.replace("</svg>", "</g></svg>")

    # remove all whitespace between elements
    svg_text = re.sub(r">\s+<", "><", svg_text)

    # Remove empty <defs/>
    svg_text = svg_text.replace("<defs/>", "")
    return svg_text


def convert_and_quantize_svg(svg_content, bins: int = 128):
    paths, _ = svgstr2paths(svg_content)
    quantized_paths = quantize_paths(paths, bins, svg_content)

    # Use paths2Drawing to get Drawing object, then write to string
    dwg = paths2Drawing(quantized_paths)
    svg_content = dwg.tostring()
    output = clean_svg(svg_content, bins)
    return output


def add_viewbox(svg_content):
    # find the width and height of the SVG using regex
    match = re.search(r'width="([^"]+)" height="([^"]+)"', svg_content)
    if match:
        width = match.group(1)
        height = match.group(2)
        viewbox_value = f"0 0 {width} {height}"
        if re.search(r'viewBox="[^"]*"', svg_content):
            svg_content = re.sub(
                r'viewBox="[^"]*"', f'viewBox="{viewbox_value}"', svg_content
            )
        else:
            svg_content = re.sub(
                r"(<svg[^>]*)",
                r'\1 viewBox="' + viewbox_value + '"',
                svg_content,
                count=1,
            )
    return svg_content


def remove_rect(svg_content):
    svg_content = re.sub(r"<rect[^>]*/>", "", svg_content)
    return svg_content


def count_curves(svg_content):
    paths, _ = svgstr2paths(svg_content)
    # count each segment in each path
    return sum(len(path) for path in paths)


def normalize(v):
    if v == 0:
        return 0
    return v / abs(v)


def dot(a, b):
    """2D dot product for complex numbers."""
    return (a.conjugate() * b).real


def chordLengthParameterize(points):
    """Assign parameter values to points using relative distances."""
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + abs(points[i] - points[i - 1]))
    u = [val / u[-1] for val in u]
    return u


# This does a greedy fit of cubic Bezier curves to a set of points
class bezier:
    @staticmethod
    def q(ctrl, t):
        return (
            ((1 - t) ** 3) * ctrl[0]
            + 3 * ((1 - t) ** 2) * t * ctrl[1]
            + 3 * (1 - t) * (t**2) * ctrl[2]
            + (t**3) * ctrl[3]
        )

    @staticmethod
    def qprime(ctrl, t):
        return (
            3 * ((1 - t) ** 2) * (ctrl[1] - ctrl[0])
            + 6 * (1 - t) * t * (ctrl[2] - ctrl[1])
            + 3 * (t**2) * (ctrl[3] - ctrl[2])
        )

    @staticmethod
    def qprimeprime(ctrl, t):
        return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (
            ctrl[3] - 2 * ctrl[2] + ctrl[1]
        )


def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)


def fitCubic(points, leftTangent, rightTangent, error):
    if len(points) == 2:
        dist = abs(points[0] - points[1]) / 3.0
        bezCurve = [
            points[0],
            points[0] + leftTangent * dist,
            points[1] + rightTangent * dist,
            points[1],
        ]
        return [bezCurve]

    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    maxError, splitPoint = computeMaxError(points, bezCurve, u)

    if maxError < error:
        return [bezCurve]

    if maxError < error**2:
        for _ in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    beziers = []
    centerTangent = normalize(points[splitPoint - 1] - points[splitPoint + 1])
    beziers += fitCubic(points[: splitPoint + 1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)
    return beziers


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    C = [[0.0, 0.0], [0.0, 0.0]]
    X = [0.0, 0.0]

    A = []
    for u in parameters:
        A1 = leftTangent * 3 * (1 - u) ** 2 * u
        A2 = rightTangent * 3 * (1 - u) * (u**2)
        A.append((A1, A2))

    for i, (point, u) in enumerate(zip(points, parameters)):
        A1, A2 = A[i]
        C[0][0] += dot(A1, A1)
        C[0][1] += dot(A1, A2)
        C[1][0] += dot(A1, A2)
        C[1][1] += dot(A2, A2)

        tmp = point - bezier.q([points[0], points[0], points[-1], points[-1]], u)
        X[0] += dot(A1, tmp)
        X[1] += dot(A2, tmp)

    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    if abs(det_C0_C1) > 1e-12:
        alpha_l = det_X_C1 / det_C0_C1
        alpha_r = det_C0_X / det_C0_C1
    else:
        alpha_l = alpha_r = 0.0

    segLength = abs(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength

    if alpha_l < epsilon or alpha_r < epsilon:
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.0)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.0)
    else:
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points) // 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = abs(bezier.q(bez, u) - point) ** 2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i
    return maxDist, splitPoint


def reparameterize(bez, points, parameters):
    return [
        newtonRaphsonRootFind(bez, point, u) for point, u in zip(points, parameters)
    ]


def newtonRaphsonRootFind(bez, point, u):
    q_u = bezier.q(bez, u)
    q1 = bezier.qprime(bez, u)
    q2 = bezier.qprimeprime(bez, u)
    numerator = dot(q_u - point, q1)
    denominator = dot(q1, q1) + dot(q_u - point, q2)
    if denominator == 0.0:
        return u
    return u - numerator / denominator


def stroke_to_bezier(svg_content, num_samples=20, maxError=1.0):
    paths, _ = svgstr2paths(svg_content)
    fitted_paths = []

    for path in paths:
        cleaned_segments = [seg for seg in path if seg.start != seg.end]
        if not cleaned_segments:
            continue

        cleaned_path = Path(*cleaned_segments)

        # Sample path into points
        points = [cleaned_path.point(i / (num_samples - 1)) for i in range(num_samples)]
        beziers = fitCurve(points, maxError=maxError)
        fitted_paths.extend([CubicBezier(b[0], b[1], b[2], b[3]) for b in beziers])
        
    fitted_paths = [Path(*fitted_paths)]
    dwg = paths2Drawing(fitted_paths)
    return dwg.tostring()



# Ramer-Douglas-Peucker (RDP) algorithm for path simplification
def rdp(points, epsilon):
    if len(points) < 3:
        return points

    x1, y1 = points[0]
    x2, y2 = points[-1]
    max_dist = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        x0, y0 = points[i]
        num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        den = ((y2 - y1)**2 + (x2 - x1)**2) ** 0.5
        dist = num / den if den != 0 else 0
        if dist > max_dist:
            index = i
            max_dist = dist

    if max_dist > epsilon:
        left = rdp(points[:index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def stroke_to_rdp(svg_content: str, epsilon=1.0):
    paths, _ = svgstr2paths(svg_content)
    fitted_paths = []

    for path in paths:
        points = []
        for seg in path:
            if not points:
                points.append((seg.start.real, seg.start.imag))
            points.append((seg.end.real, seg.end.imag))

        if len(points) < 2:
            continue

        simplified_points = rdp(points, epsilon)
        fitted_path = [
            Line(
                complex(simplified_points[i][0], simplified_points[i][1]),
                complex(simplified_points[i+1][0], simplified_points[i+1][1])
            )
            for i in range(len(simplified_points) - 1)
        ]
        fitted_paths.extend(fitted_path)
        
    fitted_paths = [Path(*fitted_paths)]
    dwg = paths2Drawing(fitted_paths)
    return dwg.tostring()

# Utilities for converting QuickDraw sketches to SVG


def quickdraw_to_svg(drawing, stroke_width=1.0, size=256):
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
