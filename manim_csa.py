"""
Manim animation for CSA Section 3.1 (Multivariate Score Quantile) using Appendix A visuals.

Render with:
    manim -pql manim_csa.py CSASection31
or higher quality:
    manim -pqh manim_csa.py CSASection31
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    NumberPlane,
    VGroup,
    Dot,
    Line,
    Polygon,
    MathTex,
    Text,
    FadeIn,
    FadeOut,
    Create,
    ReplacementTransform,
    Write,
    Transform,
    always_redraw,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    config,
)
import av
from manim.scene import scene_file_writer


# Color palette tuned for clarity.
COLOR_S1 = "#32c48d"  # green for S(1)_C
COLOR_S2 = "#ff6b6b"  # red for S(2)_C
COLOR_POLY = "#4f9dff"  # blue envelope
COLOR_FINAL = "#ffd166"  # gold final envelope


def _select_codec() -> str:
    """Pick a video codec available in the current PyAV build."""
    candidates = ["libx264", "h264", "h264_videotoolbox", "hevc_videotoolbox", "mpeg4"]
    for name in candidates:
        try:
            av.codec.Codec(name, "w")  # type: ignore[attr-defined]
            return name
        except Exception:
            continue
    return "h264"


_SELECTED_CODEC = _select_codec()


def _patched_open_partial_movie_stream(self, file_path=None):
    """Fallback to an available codec if libx264 is unavailable (e.g., conda PyAV)."""
    if file_path is None:
        file_path = self.partial_movie_files[self.renderer.num_plays]
    self.partial_movie_file_path = file_path

    fps = scene_file_writer.to_av_frame_rate(config.frame_rate)

    codec_name = _SELECTED_CODEC
    pix_fmt = "yuv420p"
    av_options = {"an": "1", "crf": "23"}

    if config.movie_file_extension == ".webm":
        codec_name = "libvpx-vp9"
        av_options["-auto-alt-ref"] = "1"
        if config.transparent:
            pix_fmt = "yuva420p"
    elif config.transparent:
        codec_name = "qtrle"
        pix_fmt = "argb"

    with av.open(file_path, mode="w") as video_container:
        stream = video_container.add_stream(codec_name, rate=fps, options=av_options)
        stream.pix_fmt = pix_fmt
        stream.width = config.pixel_width
        stream.height = config.pixel_height

        self.video_container = video_container
        self.video_stream = stream

        self.queue = scene_file_writer.Queue()
        self.writer_thread = scene_file_writer.Thread(target=self.listen_and_write, args=())
        self.writer_thread.start()


scene_file_writer.SceneFileWriter.open_partial_movie_stream = _patched_open_partial_movie_stream


def generate_scores(n1: int = 40, n2: int = 24, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic 2D scores for the two calibration splits."""
    rng = np.random.default_rng(seed)
    base = rng.multivariate_normal(
        mean=[1.4, 1.1], cov=[[0.25, 0.08], [0.08, 0.24]], size=n1 + n2
    )
    base = np.abs(base) + np.array([0.25, 0.15])
    s1 = base[:n1]
    s2 = base[n1:]
    return s1, s2


def sample_directions(k: int = 2, m: int = 6, seed: int = 11) -> np.ndarray:
    """Sample unit directions on the positive orthant of the sphere."""
    rng = np.random.default_rng(seed)
    vecs = np.abs(rng.normal(size=(m, k)))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def clip_polygon_with_halfplane(poly: list[np.ndarray], u: np.ndarray, q: float) -> list[np.ndarray]:
    """Clip a convex polygon with the half-plane u·x <= q."""
    def inside(p: np.ndarray) -> bool:
        return np.dot(u, p) <= q + 1e-8

    def intersect(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        direction = p2 - p1
        denom = np.dot(u, direction)
        if abs(denom) < 1e-8:
            return p2
        t = (q - np.dot(u, p1)) / denom
        return p1 + t * direction

    output: list[np.ndarray] = []
    for idx in range(len(poly)):
        curr = poly[idx]
        prev = poly[idx - 1]
        curr_in = inside(curr)
        prev_in = inside(prev)
        if curr_in and prev_in:
            output.append(curr)
        elif prev_in and not curr_in:
            output.append(intersect(prev, curr))
        elif not prev_in and curr_in:
            output.append(intersect(prev, curr))
            output.append(curr)
    return output


def envelope_polygon(
    directions: np.ndarray, quantiles: np.ndarray, extent: float = 4.5
) -> list[np.ndarray]:
    """Return vertices of the intersection polygon for u·x <= q constraints."""
    square = [
        np.array([-extent, -extent]),
        np.array([extent, -extent]),
        np.array([extent, extent]),
        np.array([-extent, extent]),
    ]
    poly = square
    for u, q in zip(directions, quantiles):
        poly = clip_polygon_with_halfplane(poly, u, q)
        if len(poly) == 0:
            break
    return poly


def coverage_fraction(points: np.ndarray, directions: np.ndarray, quantiles: np.ndarray) -> float:
    """Compute fraction of points inside the intersection of half-planes."""
    projections = points @ directions.T
    inside = (projections <= quantiles).all(axis=1)
    return float(np.mean(inside))


def quantiles_for_beta(points: np.ndarray, directions: np.ndarray, beta: float) -> np.ndarray:
    """Quantile per direction for a given beta."""
    projections = points @ directions.T
    return np.quantile(projections, 1 - beta, axis=0)


def beta_search(
    points: np.ndarray, directions: np.ndarray, alpha: float, tol: float = 0.01, max_iter: int = 12
) -> tuple[list[dict], np.ndarray]:
    """Binary search for beta; return iteration snapshots and final quantiles."""
    lo, hi = alpha / len(directions), alpha
    snapshots: list[dict] = []
    quantiles = quantiles_for_beta(points, directions, (lo + hi) / 2)
    for _ in range(max_iter):
        beta = 0.5 * (lo + hi)
        quantiles = quantiles_for_beta(points, directions, beta)
        cov = coverage_fraction(points, directions, quantiles)
        snapshots.append({"beta": beta, "coverage": cov, "quantiles": quantiles})
        if abs(cov - (1 - alpha)) < tol:
            break
        if cov > 1 - alpha:
            hi = beta
        else:
            lo = beta
    return snapshots, quantiles


def polygon_mobject(plane: NumberPlane, vertices: list[np.ndarray], **kwargs) -> Polygon:
    points = [plane.coords_to_point(x, y) for x, y in vertices]
    return Polygon(*points, **kwargs)


class CSASection31(Scene):
    """Visual walkthrough of CSA Section 3.1 using Appendix A cues."""

    def construct(self):
        alpha = 0.05
        plane = NumberPlane(
            x_range=[0, 4.5, 1],
            y_range=[0, 4.5, 1],
            background_line_style={"stroke_opacity": 0.25, "stroke_width": 1},
        )
        plane.add_coordinates()
        self.play(Create(plane))

        title = Text("CSA: Multivariate Score Quantile", font_size=44)
        subtitle = Text("Section 3.1 · α = 0.05", font_size=28)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(title, shift=0.2 * DOWN))
        self.play(FadeIn(subtitle, shift=0.1 * DOWN))
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle))

        s1, s2 = generate_scores()
        directions = sample_directions(m=6)

        points_s1 = VGroup(
            *[Dot(plane.coords_to_point(x, y), color=COLOR_S1, radius=0.055) for x, y in s1]
        )
        points_s2 = VGroup(
            *[Dot(plane.coords_to_point(x, y), color=COLOR_S2, radius=0.06) for x, y in s2]
        )

        legend_s1 = Text("S(1)_C used for ordering", font_size=24, color=COLOR_S1).to_edge(UP).shift(
            2.6 * LEFT
        )
        legend_s2 = Text("S(2)_C used for threshold", font_size=24, color=COLOR_S2).next_to(
            legend_s1, RIGHT, buff=0.8
        )
        self.play(FadeIn(points_s1, lag_ratio=0.05), FadeIn(points_s2, lag_ratio=0.05))
        self.play(FadeIn(legend_s1), FadeIn(legend_s2))

        split_note = Text("Split calibration scores", font_size=26).next_to(plane, DOWN, buff=0.5)
        self.play(Write(split_note))
        self.wait(0.6)

        rays = VGroup()
        ray_labels = VGroup()
        ray_len = 3.8
        for idx, u in enumerate(directions):
            ray = Line(
                plane.coords_to_point(0, 0),
                plane.coords_to_point(*(u * ray_len)),
                stroke_width=2.5,
                color="#8888ff",
            )
            rays.add(ray)
            angle_text = Text(f"u{idx+1}", font_size=20).next_to(ray.get_end(), RIGHT * 0.4 + UP * 0.1)
            ray_labels.add(angle_text)
        self.play(*[Create(r) for r in rays], *[Write(lbl) for lbl in ray_labels], run_time=1.5)

        projection_caption = Text("Project S(1)_C onto each direction", font_size=24).next_to(
            plane, LEFT, buff=0.7
        )
        self.play(Write(projection_caption))
        self.wait(0.5)

        snapshots, quantiles_final = beta_search(s1, directions, alpha=alpha)

        beta_text = MathTex(r"\beta \text{ search}", font_size=30).to_edge(RIGHT).shift(0.7 * UP)
        cov_text = MathTex(r"\text{coverage} \approx 1-\alpha", font_size=28).next_to(
            beta_text, DOWN, buff=0.3
        )
        self.play(FadeIn(beta_text), FadeIn(cov_text))

        envelope_poly = None
        coverage_tracker = None
        for step_idx, snapshot in enumerate(snapshots[:4]):  # show a few search iterations
            beta_val = snapshot["beta"]
            quantiles = snapshot["quantiles"]
            cov = snapshot["coverage"]
            verts = envelope_polygon(directions, quantiles)
            poly = polygon_mobject(
                plane,
                verts,
                color=COLOR_POLY,
                stroke_width=4,
                fill_opacity=0.1 + 0.08 * step_idx,
            )
            beta_label = MathTex(
                rf"\beta = {beta_val:.3f}", font_size=28, color=COLOR_POLY
            ).next_to(beta_text, DOWN, buff=0.8)
            cov_label = MathTex(
                rf"\text{{cov}} = {cov:.2f}", font_size=28, color=COLOR_POLY
            ).next_to(beta_label, DOWN, buff=0.2)
            if envelope_poly is None:
                self.play(FadeIn(poly), FadeIn(beta_label), FadeIn(cov_label))
            else:
                self.play(
                    ReplacementTransform(envelope_poly, poly),
                    Transform(coverage_tracker, cov_label),
                    Transform(beta_tracker, beta_label),
                )
            envelope_poly = poly
            beta_tracker = beta_label
            coverage_tracker = cov_label
            self.wait(0.5)

        if envelope_poly is None:
            return

        frontier_caption = Text("Quantile envelope A₁ defines ordering", font_size=24).next_to(
            plane, RIGHT, buff=0.7
        )
        self.play(Write(frontier_caption))
        self.wait(0.6)

        # Illustrate t(s) mapping for a held-out score.
        sample_point = s2[0]
        sample_dot = Dot(plane.coords_to_point(*sample_point), color=COLOR_S2, radius=0.08)
        self.play(FadeIn(sample_dot))

        # Identify the direction causing the largest ratio.
        ratios = (sample_point @ directions.T) / quantiles
        worst_idx = int(np.argmax(ratios))
        worst_u = directions[worst_idx]
        worst_ratio = ratios[worst_idx]
        boundary_point = (worst_u * quantiles[worst_idx]) / (worst_u @ worst_u)
        boundary_line = Line(
            plane.coords_to_point(*(sample_point)),
            plane.coords_to_point(*boundary_point),
            color=COLOR_S2,
            stroke_width=3,
        )
        t_text = MathTex(
            rf"t(s) = {worst_ratio:.2f}", font_size=30, color=COLOR_S2
        ).next_to(sample_dot, RIGHT, buff=0.3)
        self.play(Create(boundary_line), Write(t_text))
        self.wait(0.6)

        # Final adjustment using S(2)_C to obtain b_t and the final envelope.
        t_values = np.max((s2 @ directions.T) / quantiles_final, axis=1)
        bt = float(np.quantile(t_values, 1 - alpha))
        final_quantiles = quantiles_final * bt
        final_vertices = envelope_polygon(directions, final_quantiles)
        final_poly = polygon_mobject(
            plane, final_vertices, color=COLOR_FINAL, stroke_width=4, fill_opacity=0.15
        )
        final_label = MathTex(
            r"\hat{Q} = \bigcap_m H(u_m, \hat{t}\tilde{q}_m)", font_size=30, color=COLOR_FINAL
        ).to_edge(DOWN)
        bt_text = MathTex(rf"\hat{{t}} = {bt:.2f}", font_size=28, color=COLOR_FINAL).next_to(
            final_label, UP, buff=0.2
        )

        self.play(ReplacementTransform(envelope_poly, final_poly), FadeIn(bt_text))
        self.play(Write(final_label))
        self.wait(0.8)

        outro = Text("CSA multivariate quantile region with α = 0.05", font_size=26).next_to(
            plane, DOWN, buff=0.8
        )
        self.play(FadeIn(outro, shift=0.2 * UP))
        self.wait(1.2)

        self.play(
            FadeOut(points_s1),
            FadeOut(points_s2),
            FadeOut(final_poly),
            FadeOut(outro),
            FadeOut(beta_text),
            FadeOut(cov_text),
            FadeOut(beta_tracker),
            FadeOut(coverage_tracker),
            FadeOut(frontier_caption),
            FadeOut(projection_caption),
            FadeOut(split_note),
            FadeOut(ray_labels),
            FadeOut(rays),
            FadeOut(plane),
            FadeOut(sample_dot),
            FadeOut(boundary_line),
            FadeOut(t_text),
            FadeOut(final_label),
            FadeOut(bt_text),
        )
