"""
Manim animation for Sections 3.1–3.3 of the CPO paper:
- Sample K points from a conditional generative model q(c|x).
- Build the conformal region as a union of metric balls (Equation 5).
- Solve min_w max_{c in union} f(w,c) by (i) maximizing over each ball (Eq. 6),
  (ii) picking the worst of those, and (iii) updating w via Danskin/gradient.

Render (preview quality):
    manim -pql manim_cpo.py CPOSections
Higher quality:
    manim -pqh manim_cpo.py CPOSections
"""

from __future__ import annotations

import numpy as np
from manim import (
    Scene,
    NumberPlane,
    VGroup,
    Dot,
    Circle,
    Arrow,
    MathTex,
    Text,
    always_redraw,
    FadeIn,
    FadeOut,
    Create,
    Write,
    ReplacementTransform,
    Line,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)


class CPOSections(Scene):
    def construct(self):
        plane = self.intro()
        centers, radius, label, score = self.sample_and_region(plane)
        balls, region_label = self.show_union(plane, centers, radius, label, score)
        self.robust_opt(plane, centers, radius, balls, region_label)

    def intro(self):
        title = Text("Conformal Predict-Then-Optimize (Sec. 3.1–3.3)", font_size=42)
        subtitle = Text("Generative score → union of balls → robust min-max", font_size=28)
        subtitle.next_to(title, DOWN)
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=0.2 * DOWN))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-3.5, 3.5, 1],
            background_line_style={"stroke_opacity": 0.3, "stroke_width": 1},
        )
        plane.add_coordinates()
        self.play(Create(plane))
        return plane

    def sample_and_region(self, plane: NumberPlane):
        label = MathTex(r"\hat c_k \sim q(c\mid x)", font_size=34).to_edge(UP)
        score = MathTex(r"s(x,c)=\min_k d(\hat c_k,c)\quad\text{(Eq. 5)}", font_size=30)
        score.next_to(label, DOWN)
        self.play(Write(label))
        self.play(FadeIn(score, shift=0.1 * DOWN))

        rng = np.random.default_rng(12)
        centers = rng.normal(loc=[-0.5, 0.2], scale=1.5, size=(5, 2))
        dots = VGroup(
            *[Dot(plane.coords_to_point(x, y), color="#4f9dff", radius=0.06) for x, y in centers]
        )

        for d in dots:
            self.play(FadeIn(d, scale=0.2), run_time=0.15)
        self.wait(0.4)
        radius = 1.05
        return centers, radius, label, score

    def show_union(self, plane: NumberPlane, centers, radius: float, label: MathTex, score: MathTex):
        balls = VGroup(
            *[
                Circle(radius=radius, color="#4f9dff", fill_opacity=0.12, stroke_width=3).move_to(
                    plane.coords_to_point(*c)
                )
                for c in centers
            ]
        )
        union_tex = MathTex(
            r"C(x)=\bigcup_{k=1}^K B_{\hat q}(\hat c_k)\quad\text{(prediction region)}",
            font_size=30,
        ).to_edge(UP)

        self.play(FadeIn(balls, lag_ratio=0.1))
        self.play(FadeOut(score), ReplacementTransform(label, union_tex))
        helper = Text("Nonconvex region = union of balls", font_size=26).next_to(union_tex, DOWN)
        self.play(FadeIn(helper, shift=0.1 * DOWN))
        self.wait(0.6)
        return balls, union_tex

    def robust_opt(
        self,
        plane: NumberPlane,
        centers,
        radius: float,
        balls: VGroup,
        region_label: MathTex,
    ):
        decision = Dot(plane.coords_to_point(3.3, -2.0), color="#ffd166", radius=0.08)
        d_label = MathTex("w", font_size=34, color="#ffd166").next_to(decision, DOWN)
        objective = MathTex(
            r"w^*(x)=\arg\min_w \max_{c\in C(x)} f(w,c)\quad\text{(Eq. 3)}",
            font_size=30,
        ).to_edge(DOWN)
        self.play(FadeIn(decision), FadeIn(d_label))
        self.play(Write(objective))

        # Direction of increasing cost (fake gradient direction for illustration).
        grad_dir3 = np.array([-1.0, 0.5, 0.0])
        grad_dir3 = grad_dir3 / np.linalg.norm(grad_dir3)
        grad_dir2 = grad_dir3[:2]
        grad_arrow = Arrow(
            start=decision.get_center(),
            end=decision.get_center() + grad_dir3 * 1.5,
            color="#ff6b6b",
        )
        grad_label = MathTex(r"\nabla_w f", font_size=26, color="#ff6b6b").next_to(grad_arrow, UP)
        self.play(Create(grad_arrow), FadeIn(grad_label))

        # Inner maximization over union via per-ball maxima (Eq. 6).
        inner_label = MathTex(
            r"\max_{c\in C(x)} f = \max_k \max_{c\in B_{\hat q}(\hat c_k)} f",
            font_size=28,
        ).next_to(region_label, DOWN)
        self.play(FadeIn(inner_label, shift=0.1 * DOWN))

        # Helper to compute candidate point on each ball boundary along grad_dir.
        def candidate_on_ball(center_xy: np.ndarray) -> np.ndarray:
            return plane.coords_to_point(*(center_xy + radius * grad_dir2))

        candidates = []
        connectors = []
        for center_xy, ball in zip(centers, balls):
            cand_point = candidate_on_ball(center_xy)
            cand_dot = Dot(cand_point, color="#ff6b6b", radius=0.06)
            candidates.append(cand_dot)
            connectors.append(Line(ball.get_center(), cand_point, color="#ff6b6b", stroke_opacity=0.5))

        # Animate per-ball maxima and highlight the worst-case candidate.
        for line, dot, ball in zip(connectors, candidates, balls):
            self.play(FadeIn(line), FadeIn(dot), ball.animate.set_stroke(width=4), run_time=0.4)
            self.play(ball.animate.set_stroke(width=3), run_time=0.1)
        self.wait(0.3)

        # Choose the candidate with largest projection along grad_dir.
        projections = [np.dot(plane.point_to_coords(dot.get_center())[:2], grad_dir2) for dot in candidates]
        worst_idx = int(np.argmax(projections))
        worst_dot = candidates[worst_idx]
        worst_label = MathTex(r"c^*(w)", font_size=28, color="#ff6b6b").next_to(worst_dot, RIGHT)
        self.play(worst_dot.animate.scale(1.2), FadeIn(worst_label))

        # Danskin step: gradient at worst-case point moves w against cost direction.
        step_text = Text("Update w via gradient at c*(w)", font_size=26, color="#ffd166")
        step_text.next_to(objective, UP)
        self.play(FadeIn(step_text, shift=0.1 * UP))

        new_w = decision.get_center() - grad_dir3 * 0.8
        self.play(decision.animate.move_to(new_w), d_label.animate.next_to(decision, DOWN), run_time=1.0)

        # Clean up for an end frame.
        self.wait(0.6)
        self.play(
            FadeOut(inner_label),
            FadeOut(step_text),
            FadeOut(grad_arrow),
            FadeOut(grad_label),
            *[FadeOut(m) for m in connectors + candidates],
        )
        final_note = Text("Iterate until convergence (Alg. 1)", font_size=28, color="#ffd166")
        final_note.next_to(objective, UP)
        self.play(FadeIn(final_note, shift=0.2 * UP))
        self.wait(1.2)
