# robbuffet_tut

## Manim demo for conformal predict-then-optimize

`manim_cpo.py` now focuses on Sections 3.1–3.3 of the CPO paper:
- Sample K centers from q(c|x)
- Build the conformal region as a union of balls (Eq. 5)
- Inner maximization over each ball (Eq. 6) and gradient update via Danskin (Alg. 1)

Quick start:
1. Install manim (community edition) if needed: `pip install manim`
2. Render a fast preview: `manim -pql manim_cpo.py CPOSections`
3. Render higher quality: `manim -pqh manim_cpo.py CPOSections`
