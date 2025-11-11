from svgpathtools import svgstr2paths
from prepare_data import parse_viewbox, make_quantizer


# We propose two novel representations for SVG data:

## Absolute (Point-based) representation: each row is (x, y, flag)
# where x, y are absolute coordinates, and flag is 0 for "move" and 1 for "line".

## Delta (Stroke-based) representation: each row is (dx, dy, flag)
# where dx, dy are the offsets from the previous point,
# and flag is 0 for a "move" (start of new path) and 1 for a "line" (continuation of path).


class AbsolutePenPositionTokenizer:
    def __init__(self, bins=128):
        self.bins = bins
        self.vocab = {}
        self.inv_vocab = {}

        idx = 0
        for x in range(self.bins):
            for y in range(self.bins):
                self.vocab[(x, y)] = idx
                self.inv_vocab[idx] = (x, y)
                idx += 1

        for pen_token in ["MOVE", "PAD", "START", "END"]:
            self.vocab[pen_token] = idx
            self.inv_vocab[idx] = pen_token
            idx += 1

        self.pad_token_id = self.vocab["PAD"]

    def encode(self, svg_content):
        """
        Encode SVG content into a sequence of tokens.
        """
        paths, _ = svgstr2paths(svg_content)
        min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
        quantize_point = make_quantizer(min_x, max_x, min_y, max_y, self.bins)

        tokens = [self.vocab["START"]]
        move_token = self.vocab["MOVE"]

        for path in paths:
            # Move command
            start = quantize_point(path[0].start)
            tokens.append(move_token)
            tokens.append(self.vocab[(int(start.real), int(start.imag))])

            # Line segments
            for seg in path:
                end = quantize_point(seg.end)
                tokens.append(self.vocab[(int(end.real), int(end.imag))])

        tokens.append(self.vocab["END"])
        return tokens

    def decode(self, tokens, stroke_width=0.4):
        """Decode a sequence of tokens into SVG content."""
        svg_parts = [
            f'<svg viewBox="0 0 {self.bins} {self.bins}"><g stroke-width="{stroke_width}">'
        ]
        path_cmds = []

        for token in tokens:
            item = self.inv_vocab[token]
            if item == "START":
                continue
            elif item == "END":
                break
            elif item == "PAD":
                continue
            elif item == "MOVE":
                if path_cmds:
                    path_str = " ".join(path_cmds)
                    svg_parts.append(
                        f'<path d="{path_str}" stroke="black" fill="none"/>'
                    )
                    path_cmds = []
            else:
                x, y = item
                if not path_cmds:
                    path_cmds.append(f"M {x} {y}")
                else:
                    path_cmds.append(f"L {x} {y}")

        # Flush last path
        if path_cmds:
            path_str = " ".join(path_cmds)
            svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

        svg_parts.append("</g></svg>")
        return "\n".join(svg_parts)


class DeltaPenPositionTokenizer:
    def __init__(self, bins=128):
        self.bin_range = range(-bins, bins + 1)
        self.bins = bins

        self.vocab = {}
        self.inv_vocab = {}

        idx = 0
        for x in self.bin_range:
            for y in self.bin_range:
                self.vocab[(x, y)] = idx
                self.inv_vocab[idx] = (x, y)
                idx += 1

        for pen_token in ["MOVE", "PAD", "START", "END"]:
            self.vocab[pen_token] = idx
            self.inv_vocab[idx] = pen_token
            idx += 1

        self.pad_token_id = self.vocab["PAD"]

    def encode(self, svg_content):
        """
        Encode SVG content into a sequence of tokens using delta positions.
        """
        min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
        quantize_point = make_quantizer(min_x, max_x, min_y, max_y, self.bins)
        paths, _ = svgstr2paths(svg_content)
        prev = None
        move_token = self.vocab["MOVE"]

        tokens = [self.vocab["START"]]
        for path in paths:
            tokens.append(move_token)
            q_start = quantize_point(path[0].start)
            dx = q_start.real - (prev.real if prev else 0)
            dy = q_start.imag - (prev.imag if prev else 0)
            prev = q_start
            tokens.append(self.vocab[(int(dx), int(dy))])

            # Line segments
            for seg in path:
                end = quantize_point(seg.end)
                dx = end.real - prev.real
                dy = end.imag - prev.imag

                # If there is no movement skip
                if dx == 0 and dy == 0:
                    continue

                tokens.append(self.vocab[(int(dx), int(dy))])
                prev = end

        tokens.append(self.vocab["END"])
        return tokens

    def decode(self, tokens, stroke_width=0.4):
        """Decode a sequence of tokens into SVG content."""
        svg_parts = [
            f'<svg viewBox="0 0 {self.bins} {self.bins}"><g stroke-width="{stroke_width}">'
        ]
        path_cmds = []
        x, y = 0.0, 0.0

        for token in tokens:
            item = self.inv_vocab[token]
            if item == "START":
                continue
            elif item == "END":
                break
            elif item == "PAD":
                continue
            elif item == "MOVE":
                if path_cmds:
                    path_str = " ".join(path_cmds)
                    svg_parts.append(
                        f'<path d="{path_str}" stroke="black" fill="none"/>'
                    )
                    path_cmds = []
            else:
                dx, dy = item
                x += dx
                y += dy

                if not path_cmds:
                    path_cmds.append(f"M {x} {y}")
                else:
                    path_cmds.append(f"L {x} {y}")

        # Flush last path
        if path_cmds:
            path_str = " ".join(path_cmds)
            svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

        svg_parts.append("</g></svg>")
        return "\n".join(svg_parts)


class AbsoluteBezierPenPositionTokenizer:

    def __init__(self, bins=128):
        self.bins = bins
        self.vocab = {}
        self.inv_vocab = {}

        idx = 0
        for x in range(self.bins):
            for y in range(self.bins):
                self.vocab[(x, y)] = idx
                self.inv_vocab[idx] = (x, y)
                idx += 1

        for pen_token in ["MOVE", "PAD", "START", "END"]:
            self.vocab[pen_token] = idx
            self.inv_vocab[idx] = pen_token
            idx += 1

        self.pad_token_id = self.vocab["PAD"]

    def encode(self, svg_content):
        """
        Encode SVG content into a sequence of tokens using Bezier curves.
        """
        paths, _ = svgstr2paths(svg_content)
        min_x, max_x, min_y, max_y = parse_viewbox(svg_content)
        quantize_point = make_quantizer(min_x, max_x, min_y, max_y, self.bins)

        tokens = [self.vocab["START"]]
        for path in paths:
            tokens.append(self.vocab["MOVE"])
            q_start = quantize_point(path[0].start)
            tokens.append(self.vocab[(int(q_start.real), int(q_start.imag))])

            # Line segments
            for seg in path:
                q_ctrl1 = quantize_point(seg.control1)
                q_ctrl2 = quantize_point(seg.control2)
                q_end = quantize_point(seg.end)

                tokens.append(self.vocab[(int(q_ctrl1.real), int(q_ctrl1.imag))])
                tokens.append(self.vocab[(int(q_ctrl2.real), int(q_ctrl2.imag))])
                tokens.append(self.vocab[(int(q_end.real), int(q_end.imag))])

        tokens.append(self.vocab["END"])
        return tokens

    def decode(self, tokens, stroke_width=0.4):
        """Decode a sequence of tokens into SVG content."""
        svg_parts = [
            f'<svg viewBox="0 0 {self.bins} {self.bins}"><g stroke-width="{stroke_width}">'
        ]
        path_cmds = []
        curve_cmds = []

        for token in tokens:
            item = self.inv_vocab[token]
            if item == "START":
                continue
            elif item == "END":
                break
            elif item == "PAD":
                continue
            elif item == "MOVE":
                if path_cmds:
                    path_str = " ".join(path_cmds)
                    svg_parts.append(
                        f'<path d="{path_str}" stroke="black" fill="none"/>'
                    )
                    path_cmds = []
                    curve_cmds = []
            else:
                x, y = item
                if not path_cmds:
                    path_cmds.append(f"M {x} {y}")
                else:
                    curve_cmds.append((x, y))

                    if len(curve_cmds) == 3:
                        c1, c2, end = curve_cmds
                        path_cmds.append(
                            f"C {c1[0]} {c1[1]}, {c2[0]} {c2[1]}, {end[0]} {end[1]}"
                        )
                        curve_cmds = []

        # Flush last path
        if path_cmds:
            path_str = " ".join(path_cmds)
            svg_parts.append(f'<path d="{path_str}" stroke="black" fill="none"/>')

        svg_parts.append("</g></svg>")
        return "\n".join(svg_parts)
