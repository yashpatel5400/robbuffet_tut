# robbuffet_tut

## Manim demo for conformal predict-then-optimize

`manim_cpo.py` contains a short animation that illustrates:
- Sampling from a conditional generative model
- Building a conformal prediction region as a union of balls
- Choosing a robust decision that minimizes the worst-case loss

Quick start:
1. Install manim (community edition) if needed: `pip install manim`
2. Render a fast preview: `manim -pql manim_cpo.py CPOStory`
3. Render higher quality: `manim -pqh manim_cpo.py CPOStory`
