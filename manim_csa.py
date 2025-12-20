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
    VMobject,
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
    UL,
    UR,
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


def generate_scores(n1: int = 28, n2: int = 16, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
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
    vecs = vecs / norms
    # Force u_1 to be the y-axis and u_5 to be the x-axis.
    if m >= 1:
        vecs[0] = np.array([0.0, 1.0])
    if m >= 5:
        vecs[4] = np.array([1.0, 0.0])
    return vecs


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

    def projection_marks(
        self, points: np.ndarray, direction: np.ndarray, plane: NumberPlane, color: str
    ) -> VGroup:
        projections = points @ direction
        return VGroup(
            *[
                Dot(
                    plane.coords_to_point(*(direction * p)),
                    color=color,
                    radius=0.04,
                    fill_opacity=0.9,
                )
                for p in projections
            ]
        )

    def projection_anim(
        self, points: np.ndarray, direction: np.ndarray, plane: NumberPlane, color: str
    ) -> tuple[VGroup, VGroup]:
        """Return projection targets and animating lines for moving dots onto a direction."""
        target_points = VGroup()
        lines = VGroup()
        for x, y in points:
            proj_len = np.dot(np.array([x, y]), direction)
            proj_xy = direction * proj_len
            target = Dot(plane.coords_to_point(*proj_xy), color="#4f9dff", radius=0.04)
            line = Line(
                plane.coords_to_point(x, y),
                plane.coords_to_point(*proj_xy),
                color=color,
                stroke_width=2,
                stroke_opacity=0.6,
            )
            target_points.add(target)
            lines.add(line)
        return target_points, lines

    def boundary_line(self, direction: np.ndarray, q: float, plane: NumberPlane, color: str) -> Line:
        x_int = np.array([q / max(direction[0], 1e-6), 0.0])
        y_int = np.array([0.0, q / max(direction[1], 1e-6)])
        return Line(
            plane.coords_to_point(*x_int),
            plane.coords_to_point(*y_int),
            color=color,
            stroke_width=3,
            stroke_opacity=0.8,
        )

    def construct(self):
        alpha = 0.05
        beta_text_color = "#6c6cff"
        title = Text("CSA: multivariate score quantile", font_size=44)
        subtitle = Text("Predictors → split conformal → envelope", font_size=26)
        subtitle.next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(title, shift=0.2 * DOWN))
        self.play(FadeIn(subtitle, shift=0.1 * DOWN))
        self.wait(0.6)
        self.play(FadeOut(title), FadeOut(subtitle))

        plane = NumberPlane(
            x_range=[0, 4.5, 1],
            y_range=[0, 4.5, 1],
            background_line_style={"stroke_opacity": 0.25, "stroke_width": 1},
            axis_config={
                "color": COLOR_S1,
            },
            y_axis_config={
                "color": COLOR_S2,
            },
        )
        plane.set_width(config.frame_width * 4.8)
        plane.set_height(config.frame_height * 0.8)
        plane.shift(0.1 * DOWN)
        plane.add_coordinates()
        self.play(Create(plane))

        s1, s2 = generate_scores()
        directions = sample_directions(m=5)

        predictor_1 = MathTex(
            r"C_1(x)=\{y: s_1(x,y)\le q_1\}", font_size=30, color=COLOR_S1
        )
        predictor_2 = MathTex(
            r"C_2(x)=\{y: s_2(x,y)\le q_2\}", font_size=30, color=COLOR_S2
        )
        predictors = VGroup(predictor_1, predictor_2).arrange(RIGHT, buff=1.4).to_edge(UP, buff=0.25)
        split_note = Text("Split conformal per predictor", font_size=26).next_to(
            predictors, DOWN, buff=0.2
        )
        self.play(Write(predictors))
        self.play(FadeIn(split_note))

        points_s1 = VGroup(
            *[Dot(plane.coords_to_point(x, y), color="#4f9dff", radius=0.055) for x, y in s1]
        )
        s1_caption = MathTex(
            r"\mathcal{S}_{C}^{(1)} \text{ scores (ordering)}", font_size=24, color=COLOR_S1
        ).to_corner(
            UL, buff=0.35
        )
        for dot in points_s1:
            self.play(FadeIn(dot, scale=0.5), run_time=0.08)
        self.play(FadeIn(s1_caption))
        self.wait(0.2)
        self.play(FadeOut(predictors), FadeOut(split_note), run_time=0.4)

        rays = VGroup()
        ray_labels = VGroup()
        ray_len = 3.6
        for idx, u in enumerate(directions):
            ray = Line(
                plane.coords_to_point(0, 0),
                plane.coords_to_point(*(u * ray_len)),
                stroke_width=2.5,
                color="#8888ff",
            )
            rays.add(ray)
            lbl = MathTex(rf"u_{{{idx+1}}}", font_size=20).next_to(
                ray.get_end(), RIGHT * 0.4 + UP * 0.1
            )
            ray_labels.add(lbl)
        self.play(*[Create(r) for r in rays], *[Write(lbl) for lbl in ray_labels], run_time=1.4)

        snapshots, quantiles = beta_search(s1, directions, alpha=alpha, tol=0.003, max_iter=14)
        beta_star = snapshots[-1]["beta"] if snapshots else alpha / len(directions)

        boundary_lines = VGroup()
        thresh_labels = VGroup()
        quantile_dots = VGroup()
        quantile_points = []
        for idx, u in enumerate(directions):
            highlighted = rays[idx].copy().set_color(COLOR_POLY).set_stroke(width=4)
            self.play(Transform(rays[idx], highlighted), run_time=0.35)

            proj_targets, proj_lines = self.projection_anim(s1, u, plane, color=COLOR_S1)
            proj_group = VGroup(proj_lines, proj_targets)
            self.play(Create(proj_lines), run_time=0.5)
            self.play(
                *[Transform(points_s1[i], proj_targets[i]) for i in range(len(points_s1))],
                run_time=0.7,
            )

            q_dir = quantiles[idx]
            q_line = self.boundary_line(u, q_dir, plane, color=COLOR_POLY)
            q_point = u * q_dir
            q_dot = Dot(plane.coords_to_point(*q_point), color=COLOR_POLY, radius=0.08)
            self.play(Create(q_line), FadeIn(q_dot), run_time=0.5)
            boundary_lines.add(q_line)
            quantile_dots.add(q_dot)
            quantile_points.append(q_point)

            self.play(FadeOut(proj_group, run_time=0.4))
            self.wait(0.1)

        contour_poly = None
        order = [0, 2, 1, 3, 4]  # u1 -> u3 -> u2 -> u4 -> u5 (clockwise)
        if len(quantile_points) >= 2:
            ordered_pts = [quantile_points[i] for i in order if i < len(quantile_points)]
            if len(ordered_pts) >= 2:
                contour_poly = VMobject(color=COLOR_FINAL, stroke_width=3)
                contour_poly.set_points_as_corners(
                    [plane.coords_to_point(x, y) for x, y in ordered_pts]
                )
                self.play(Create(contour_poly))
                self.wait(0.2)

        # Step 8: expand/shrink the contour to illustrate search toward 1-α coverage.
        origin = plane.coords_to_point(0, 0)
        inflate_poly = None
        if contour_poly:
            inflate_poly = contour_poly.copy().scale(1.1, about_point=origin)
            self.play(ReplacementTransform(contour_poly, inflate_poly), run_time=0.6)
        else:
            fallback_vertices = envelope_polygon(directions, quantiles)
            inflate_poly = polygon_mobject(
                plane, fallback_vertices, color=COLOR_POLY, stroke_width=3, fill_opacity=0.08
            ).scale(1.1, about_point=origin)
            self.play(Create(inflate_poly), run_time=0.6)

        # Introduce S(2)_C and final adjustment.
        points_s2 = VGroup(
            *[Dot(plane.coords_to_point(x, y), color=COLOR_S2, radius=0.06) for x, y in s2]
        )
        s2_caption = MathTex(
            r"\mathcal{S}_{C}^{(2)} \text{ scores (adjustment)}", font_size=24, color=COLOR_S2
        ).next_to(
            s1_caption, DOWN, buff=0.2
        )
        self.play(FadeIn(points_s2, lag_ratio=0.05), FadeIn(s2_caption), run_time=1.0)
        self.wait(0.2)

        t_values = np.max((s2 @ directions.T) / quantiles, axis=1)
        bt = float(np.quantile(t_values, 1 - alpha))
        final_quantiles = quantiles * bt
        final_poly = (
            inflate_poly.copy()
            .scale(bt, about_point=origin)
            .set_stroke(width=4, color=COLOR_FINAL)
            .set_fill(opacity=0.18)
            if inflate_poly
            else polygon_mobject(
                plane,
                envelope_polygon(directions, final_quantiles),
                color=COLOR_FINAL,
                stroke_width=4,
                fill_opacity=0.18,
            )
        )

        bt_text = MathTex(rf"\hat{{t}}={bt:.2f}", font_size=28, color=COLOR_FINAL)
        self.play(ReplacementTransform(inflate_poly, final_poly), FadeIn(bt_text))
        self.wait(0.6)

        final_label = MathTex(
            r"\hat{Q} = \bigcap_m H(u_m, \hat{t}\tilde{q}_m)", font_size=30, color=COLOR_FINAL
        ).to_corner(UR, buff=0.6)
        coverage_label = MathTex(
            r"|S^{(2)}_C \cap \hat{Q}| \approx 1-\alpha", font_size=26, color=COLOR_FINAL
        ).next_to(final_label, DOWN, buff=0.2)
        self.play(Write(final_label), Write(coverage_label))
        self.wait(1.0)

        self.play(
            FadeOut(points_s1),
            FadeOut(points_s2),
            FadeOut(final_poly),
            FadeOut(thresh_labels),
            FadeOut(quantile_dots),
            FadeOut(contour_poly) if contour_poly else FadeOut(VGroup()),
            FadeOut(ray_labels),
            FadeOut(rays),
            FadeOut(boundary_lines),
            FadeOut(s1_caption),
            FadeOut(s2_caption),
            FadeOut(split_note),
            FadeOut(predictor_1),
            FadeOut(predictor_2),
            FadeOut(bt_text),
            FadeOut(final_label),
            FadeOut(coverage_label),
            FadeOut(plane),
        )
        # Ensure a clean ending frame with nothing lingering.
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=0.5)
