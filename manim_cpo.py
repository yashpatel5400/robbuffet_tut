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
    Transform,
    Rotate,
    Line,
    SurroundingRectangle,
    ORIGIN,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
)


class CPOSections(Scene):
    def construct(self):
        plane = self.intro()
        centers1, _, label, score, truth, sample_dots1, truth_group1, labels1 = self.sample_and_region(plane)
        union_anchor = label.copy().set_opacity(0.001)
        self.add(union_anchor)
        min_radius, min_line, dots1, truth_group, all_lines = self.show_score_demo(
            plane, centers1, truth, sample_dots1, score
        )
        centers2, template_line, dots2, labels2, new_label = self.resample_with_template(
            plane, min_radius, min_line, sample_dots1, truth_group1, all_lines, labels1, label, score
        )
        balls, region_label = self.show_union(
            plane, centers2, min_radius, union_anchor, score, template_line, dots2, labels2, new_label
        )
        self.robust_opt(plane, centers2, min_radius, balls, region_label)

    def intro(self):
        title = Text("Conformal Predict-Then-Optimize", font_size=42)
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
        label = MathTex(r"\hat c_k \sim q(c\mid x)", font_size=34).to_corner(UL).shift(DOWN * 0.2)
        score = MathTex(r"s(x,c)=\min_k d(\hat c_k,c)", font_size=30).next_to(
            label, DOWN, aligned_edge=LEFT
        )
        self.score_anchor = score
        self.play(Write(label))
        self.play(FadeIn(score, shift=0.1 * DOWN))

        truth = Dot(plane.coords_to_point(0.2, 0.2), color="#ff6b6b", radius=0.08)
        truth_label = MathTex("c", font_size=32, color="#ff6b6b").next_to(truth, UP)
        self.play(FadeIn(truth), FadeIn(truth_label))

        rng = np.random.default_rng(12)
        centers = rng.normal(loc=[0.0, 0.0], scale=1.5, size=(5, 2))
        dots = VGroup(
            *[Dot(plane.coords_to_point(x, y), color="#4f9dff", radius=0.06) for x, y in centers]
        )

        labels = VGroup()
        for idx, dot in enumerate(dots, start=1):
            lbl = MathTex(rf"\hat c_{idx}", font_size=26, color="#4f9dff").next_to(dot, UP, buff=0.12)
            self.play(FadeIn(dot, scale=0.2), FadeIn(lbl, scale=0.2), run_time=0.2)
            labels.add(lbl)
        sample_dots = VGroup(*dots)
        truth_group = VGroup(truth, truth_label)
        radius = None  # placeholder; actual radius derived from min distance
        return centers, radius, label, score, truth, sample_dots, truth_group, labels

    def show_score_demo(self, plane: NumberPlane, centers, truth: Dot, sample_dots: VGroup, score: MathTex):
        # Sequentially show distances to the truth and label them.
        kept_lines = VGroup()
        lengths = []
        truth_coords = np.array(plane.point_to_coords(truth.get_center())[:2])
        for idx, center_xy in enumerate(centers, start=1):
            start_pt = plane.coords_to_point(*center_xy)
            line = Line(start_pt, truth.get_center(), color="#ff6b6b")
            mid = (start_pt + truth.get_center()) / 2
            # Offset the label perpendicular to the line to avoid overlap.
            direction = truth.get_center() - start_pt
            perp = np.array([-direction[1], direction[0], 0.0])
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            dist_label = MathTex(
                rf"d(\hat c_{idx}, c)",
                font_size=26,
                color="#ff6b6b",
            )
            # Position distance text to the left under the score equation.
            if hasattr(self, "score_anchor"):
                dist_label.next_to(self.score_anchor, DOWN, aligned_edge=LEFT, buff=0.4)
            else:
                dist_label.move_to(mid + 0.25 * perp + 0.1 * UP)
            self.play(Create(line), FadeIn(dist_label), run_time=0.9)
            self.wait(0.35)
            self.play(line.animate.set_opacity(0.2), FadeOut(dist_label), run_time=0.4)
            kept_lines.add(line)
            lengths.append(np.linalg.norm(np.array(center_xy) - truth_coords))
        # Highlight the shortest distance.
        min_idx = int(np.argmin(lengths))
        min_line = kept_lines[min_idx]
        self.play(min_line.animate.set_opacity(1.0).set_stroke(width=5, color="#ff6b6b"))
        min_radius = lengths[min_idx]

        # Highlight the score already on screen instead of duplicating.
        self.play(score.animate.set_color("#ff6b6b"))
        self.wait(0.5)
        return min_radius, min_line, sample_dots, VGroup(truth), kept_lines

    def resample_with_template(
        self,
        plane: NumberPlane,
        min_radius: float,
        min_line: Line,
        sample_dots: VGroup,
        truth_group: VGroup,
        all_lines: VGroup,
        old_labels: VGroup,
        label: MathTex,
        score: MathTex,
    ):
        # Move the minimal line to the side and clear the old points/lines.
        other_lines = VGroup(*[l for l in all_lines if l is not min_line])
        fades = [FadeOut(sample_dots), FadeOut(truth_group), FadeOut(other_lines), FadeOut(old_labels)]
        if score in self.mobjects:
            fades.append(FadeOut(score))
        if label in self.mobjects:
            fades.append(FadeOut(label))
        target_line = Line(ORIGIN, ORIGIN + RIGHT * min_line.get_length(), color="#ff6b6b", stroke_width=5)
        target_line.shift(plane.coords_to_point(3.5, 2.4) - target_line.get_start())
        self.play(*fades, Transform(min_line, target_line), run_time=0.9)
        template_line = min_line
        template_line.set_opacity(1.0)

        new_label = MathTex(r"\hat c_k \sim q(C \mid x')", font_size=34).to_corner(UL).shift(DOWN * 0.2)
        self.play(FadeIn(new_label, shift=0.1 * DOWN))

        rng = np.random.default_rng(21)
        centers = rng.normal(loc=[-0.3, -0.1], scale=1.6, size=(5, 2))
        dots = VGroup()
        labels = VGroup()
        for idx, (x, y) in enumerate(centers, start=1):
            d = Dot(plane.coords_to_point(x, y), color="#4f9dff", radius=0.06)
            self.play(FadeIn(d, scale=0.2), run_time=0.25)
            lbl = MathTex(rf"\hat c_{idx}", font_size=26, color="#4f9dff").next_to(d, UP, buff=0.12)
            self.add(lbl)
            dots.add(d)
            labels.add(lbl)
        return centers, template_line, dots, labels, new_label

    def show_union(
        self,
        plane: NumberPlane,
        centers,
        radius: float,
        label: MathTex,
        score: MathTex,
        template_line: Line,
        dots: VGroup,
        labels: VGroup,
        new_label: MathTex,
    ):
        # Use the saved radius to spawn lines at each new center, rotate them into balls, then show the union.
        radius_vec = plane.coords_to_point(radius, 0) - plane.coords_to_point(0, 0)
        radius_screen = np.linalg.norm(radius_vec)
        anchored_lines = VGroup()
        balls = VGroup()
        for c in centers:
            start = plane.coords_to_point(*c)
            end = start + radius_vec
            line = template_line.copy().put_start_and_end_on(start, end).set_opacity(0.7)
            anchored_lines.add(line)
            balls.add(
                Circle(radius=radius_screen, color="#4f9dff", fill_opacity=0.12, stroke_width=3).move_to(start)
            )

        self.play(*[FadeIn(line) for line in anchored_lines], run_time=0.6)
        self.play(
            *[Rotate(line, angle=2 * np.pi, about_point=line.get_start()) for line in anchored_lines],
            FadeIn(balls, lag_ratio=0.1),
            run_time=1.6,
        )
        self.play(anchored_lines.animate.set_opacity(0.15), FadeOut(template_line))

        union_tex = MathTex(
            r"C(x)=\bigcup_{k=1}^K B_{\hat q}(\hat c_k)",
            font_size=30,
        ).to_corner(UR).shift(LEFT * 1.2)

        fades = []
        if score in self.mobjects:
            fades.append(FadeOut(score))
        if new_label in self.mobjects:
            fades.append(FadeOut(new_label))
        fades.extend([FadeOut(dots), FadeOut(labels)])
        self.play(*fades, ReplacementTransform(label, union_tex))
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
        d_label = MathTex("w^{(0)}", font_size=34, color="#ffd166").next_to(decision, DOWN)
        self.play(FadeIn(decision), FadeIn(d_label))

        # Show Danskin definitions.
        # Remove earlier equations before showing Danskin step.
        self.play(FadeOut(region_label), run_time=0.6)

        phi_def = MathTex(
            r"\phi(w):=\max_{\hat c \in \mathcal{C}(x)} f(w,\hat c)",
            font_size=28,
        ).to_corner(UR).shift(LEFT * 0.2, DOWN * 0.2)
        grad_def = MathTex(
            r"\nabla_w \phi(w)=\nabla_w f(w,c^*)\\ c^*:=\max_{\hat c \in \mathcal{C}(x)} f(w,\hat c)",
            font_size=28,
        ).next_to(phi_def, DOWN, aligned_edge=RIGHT)
        self.play(FadeIn(phi_def, shift=0.1 * DOWN))
        self.play(FadeIn(grad_def, shift=0.1 * DOWN))

        # Direction of increasing cost (fake gradient direction for illustration).
        grad_dir3 = np.array([-1.0, 0.5, 0.0])
        grad_dir3 = grad_dir3 / np.linalg.norm(grad_dir3)
        grad_dir2 = grad_dir3[:2]

        rng = np.random.default_rng(7)

        # Inner max decomposition equation.
        self.wait(0.6)
        inner_eq = MathTex(
            r"\max_{\hat c \in \mathcal{C}(x)} f(w,\hat c)"
            r"= \max_{k} \max_{\hat c \in \mathcal{B}_{\hat q}(\hat c_{k})} f(w,\hat c)",
            font_size=26,
        ).next_to(grad_def, DOWN, aligned_edge=RIGHT)
        self.play(FadeIn(inner_eq, shift=0.1 * DOWN))
        inner_highlight = SurroundingRectangle(inner_eq, color="#ffd166", buff=0.15, stroke_width=3)
        self.play(Create(inner_highlight))

        # Per-ball maxima sequentially.
        candidates = []
        ck_labels = []
        for idx, (center_xy, ball) in enumerate(zip(centers, balls), start=1):
            frac = 0.45 + 0.35 * rng.random()
            cand_xy = center_xy + frac * radius * grad_dir2
            cand_point = plane.coords_to_point(*cand_xy)
            cand_dot = Dot(cand_point, color="#ff6b6b", radius=0.06)
            candidates.append(cand_dot)
            ck_label = MathTex(rf"c^*_{{{idx}}}", font_size=24, color="#ff6b6b").next_to(cand_dot, RIGHT)
            ck_labels.append(ck_label)
            self.play(ball.animate.set_stroke(width=5, color="#ffd166"), run_time=0.6)
            self.wait(1.0)
            self.play(FadeIn(cand_dot), FadeIn(ck_label), run_time=0.6)
            self.play(ball.animate.set_stroke(width=3, color="#4f9dff"), run_time=0.2)
        self.wait(0.4)

        # Choose the candidate with largest projection along grad_dir.
        projections = [np.dot(plane.point_to_coords(dot.get_center())[:2], grad_dir2) for dot in candidates]
        worst_idx = int(np.argmax(projections))
        worst_dot = candidates[worst_idx]
        worst_label = MathTex(r"c^*", font_size=28, color="#ffd166").next_to(worst_dot, RIGHT)
        fade_rest = [FadeOut(m) for i, m in enumerate(candidates) if i != worst_idx]
        fade_rest += [FadeOut(m) for i, m in enumerate(ck_labels) if i != worst_idx]
        self.play(
            worst_dot.animate.scale(1.2).set_color("#ffd166"),
            FadeIn(worst_label),
            *fade_rest,
            run_time=0.6,
        )

        # Small gradient arrow near w^{(0)} and projected step text.
        grad_arrow = Arrow(
            start=decision.get_center() + 0.15 * LEFT,
            end=decision.get_center() + 0.4 * LEFT + 0.15 * UP,
            color="#ff6b6b",
            buff=0.0,
            stroke_width=4,
        )
        update_eq = MathTex(
            r"w^{(t)} \gets \Pi_{\mathcal{W}}\!\big(w^{(t+1)} - \eta\, \nabla_{w} f(w^{(t+1)}, c^{*})\big)",
            font_size=24,
            color="#ffd166",
        ).next_to(grad_arrow, RIGHT, buff=0.2).shift(0.1 * UP)
        self.play(Create(grad_arrow), FadeIn(update_eq), run_time=0.8)

        # Move w^{(0)} along the gradient direction to w^{(1)}.
        step_vec = grad_arrow.get_end() - grad_arrow.get_start()
        w1_label = MathTex("w^{(1)}", font_size=34, color="#ffd166").next_to(decision, DOWN)
        self.play(
            decision.animate.move_to(decision.get_center() + step_vec),
            ReplacementTransform(d_label, w1_label),
            run_time=0.9,
        )
        d_label = w1_label

        # Remove c* candidate visuals after the first step.
        self.play(*[FadeOut(m) for m in candidates + ck_labels], FadeOut(worst_label), run_time=0.4)

        # Clear equations before iterative updates.
        fades = [FadeOut(phi_def), FadeOut(grad_def), FadeOut(inner_eq), FadeOut(inner_highlight), FadeOut(update_eq), FadeOut(grad_arrow)]
        self.play(*fades, run_time=0.6)

        # Iterative updates with wobbling steps and label toggling.
        iterate_text = Text("Iterate until convergence", font_size=28, color="#ffd166").to_edge(DOWN)
        if iterate_text not in self.mobjects:
            self.play(FadeIn(iterate_text, shift=0.2 * UP))
        step_offsets = [np.array([-0.3, 0.15, 0.0]), np.array([0.2, -0.25, 0.0]), np.array([0.15, 0.2, 0.0])]
        step_num = 2
        for _, off in enumerate(step_offsets):
            new_pos = decision.get_center() + off
            new_label = MathTex(rf"w^{{({step_num})}}", font_size=34, color="#ffd166").next_to(new_pos, DOWN)
            self.play(decision.animate.move_to(new_pos), ReplacementTransform(d_label, new_label), run_time=0.7)
            d_label = new_label
            step_num += 1

        # Clean up for an end frame.
        self.wait(0.6)
        self.wait(1.2)
