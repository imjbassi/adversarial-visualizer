#!/usr/bin/env python3
"""Render an animated demo video of a PGD attack against ResNet-18.

Runs the attack step by step, records every intermediate adversarial image
and probability vector, and renders a dark-themed animation: original /
perturbation / adversarial panels, live top-5 prediction bars, and a
confidence curve that marks the moment the prediction flips.

Headless-friendly (uses the Agg backend). Requires imageio + imageio-ffmpeg
for MP4 output (``pip install imageio imageio-ffmpeg``); GIF output needs
only imageio + Pillow.

Examples:
    python scripts/make_demo_video.py --image dog.jpg --out demo.mp4
    python scripts/make_demo_video.py --image https://... --out demo.gif --width 640
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model, build_transform
from utils.image_utils import load_image_from_url, load_image_from_file

# ---------------------------------------------------------------- palette

BG = '#0d1117'
PANEL = '#161b22'
FG = '#e6edf3'
MUTED = '#8b949e'
CYAN = '#22d3ee'
MAGENTA = '#f472b6'
GREEN = '#4ade80'
RED = '#f87171'
AMBER = '#fbbf24'


def record_pgd_attack(model, image, label, epsilon=0.03, alpha=0.0012,
                      iters=150, momentum=0.9):
    """Run PGD step by step, returning a snapshot after every iteration.

    The L-inf budget ramps linearly from 0 to `epsilon` across the run, so
    the animation shows the perturbation growing and the confidence decaying
    gradually instead of collapsing in the first few frames.
    """
    ori = image.clone().detach()
    x = ori.clone().detach()
    grad_accum = torch.zeros_like(x)

    snapshots = []

    def snap(it, tensor):
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        delta = (tensor - ori).detach()
        snapshots.append({
            'iter': it,
            'image': tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().copy(),
            'probs': probs,
            'l2': float(delta.norm()),
            'linf': float(delta.abs().max()),
        })

    snap(0, x)
    for i in range(iters):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), label)
        model.zero_grad()
        loss.backward()
        grad = x.grad / (x.grad.abs().mean() + 1e-12)
        grad_accum = momentum * grad_accum + grad
        adv = x + alpha * grad_accum.sign()
        eps_i = epsilon * (i + 1) / iters
        eta = torch.clamp(adv - ori, -eps_i, eps_i)
        x = torch.clamp(ori + eta, 0, 1).detach()
        snap(i + 1, x)
    return snapshots


def styled_axes(fig, rect, title=None):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=FG, fontsize=11, pad=8)
    return ax


def short_name(name, limit=22):
    name = name.split(',')[0]
    return name if len(name) <= limit else name[:limit - 1] + '…'


class DemoRenderer:
    """Renders one frame of the demo animation at a time."""

    def __init__(self, snapshots, categories, orig_class, epsilon, width=1280):
        self.snaps = snapshots
        self.cats = categories
        self.orig_class = orig_class
        self.epsilon = epsilon
        self.orig_img = snapshots[0]['image']
        self.final_class = int(np.argmax(snapshots[-1]['probs']))

        # Iteration where the prediction first flips (if it does)
        self.flip_iter = None
        for s in snapshots:
            if int(np.argmax(s['probs'])) != orig_class:
                self.flip_iter = s['iter']
                break

        self.fig = plt.figure(figsize=(width / 100, width * 9 / 16 / 100),
                              dpi=100)
        self.fig.patch.set_facecolor(BG)
        self._build_layout()

    def _build_layout(self):
        fig = self.fig
        fig.text(0.5, 0.955, 'Adversarial Attack Visualizer', color=FG,
                 fontsize=19, fontweight='bold', ha='center')
        self.subtitle = fig.text(
            0.5, 0.912,
            'Projected Gradient Descent vs. ImageNet ResNet-18',
            color=MUTED, fontsize=11.5, ha='center')

        self.ax_orig = styled_axes(fig, [0.050, 0.415, 0.255, 0.435])
        self.ax_pert = styled_axes(fig, [0.373, 0.415, 0.255, 0.435])
        self.ax_adv = styled_axes(fig, [0.695, 0.415, 0.255, 0.435])
        for ax in (self.ax_orig, self.ax_pert, self.ax_adv):
            ax.set_xticks([])
            ax.set_yticks([])

        fig.text(0.339, 0.62, '+', color=CYAN, fontsize=26,
                 fontweight='bold', ha='center', va='center')
        fig.text(0.6615, 0.62, '=', color=CYAN, fontsize=26,
                 fontweight='bold', ha='center', va='center')

        self.ax_bars = styled_axes(fig, [0.115, 0.075, 0.320, 0.235])
        self.ax_curve = styled_axes(fig, [0.545, 0.075, 0.405, 0.235])

        self.hud = fig.text(0.95, 0.955, '', color=CYAN, fontsize=11,
                            family='monospace', ha='right')
        self.metrics = fig.text(0.05, 0.955, '', color=MUTED, fontsize=10,
                                family='monospace', ha='left')

    def render(self, snap, final=False):
        cats, orig_class = self.cats, self.orig_class
        probs = snap['probs']
        pred = int(np.argmax(probs))
        fooled = pred != orig_class

        # --- image panels
        self.ax_orig.clear()
        self.ax_orig.imshow(self.orig_img)
        self.ax_orig.set_xticks([]), self.ax_orig.set_yticks([])
        p0 = self.snaps[0]['probs']
        self.ax_orig.set_title('Original', color=FG, fontsize=12, pad=8)
        self.ax_orig.set_xlabel(
            f"{short_name(cats[orig_class])}  {p0[orig_class]:.1%}",
            color=GREEN, fontsize=10.5)

        self.ax_pert.clear()
        diff = np.abs(snap['image'] - self.orig_img).mean(axis=2)
        self.ax_pert.imshow(diff, cmap='inferno',
                            vmin=0, vmax=max(self.epsilon, 1e-6))
        self.ax_pert.set_xticks([]), self.ax_pert.set_yticks([])
        self.ax_pert.set_title('Perturbation', color=FG, fontsize=12, pad=8)
        self.ax_pert.set_xlabel(
            f"L∞ = {snap['linf']:.4f}   (budget ε = {self.epsilon})",
            color=AMBER, fontsize=10.5)

        self.ax_adv.clear()
        self.ax_adv.imshow(np.clip(snap['image'], 0, 1))
        self.ax_adv.set_xticks([]), self.ax_adv.set_yticks([])
        self.ax_adv.set_title('Adversarial', color=FG, fontsize=12, pad=8)
        self.ax_adv.set_xlabel(
            f"{short_name(cats[pred])}  {probs[pred]:.1%}",
            color=RED if fooled else GREEN, fontsize=10.5,
            fontweight='bold' if fooled else 'normal')
        for spine in self.ax_adv.spines.values():
            spine.set_color(RED if fooled else '#30363d')
            spine.set_linewidth(2.2 if fooled else 1.0)

        if final and fooled:
            self.ax_adv.text(0.5, 0.06, 'MISCLASSIFIED', color='white',
                             fontsize=13, fontweight='bold', ha='center',
                             transform=self.ax_adv.transAxes,
                             bbox=dict(facecolor=RED, alpha=0.85,
                                       edgecolor='none', pad=5))

        # --- top-5 bars
        self.ax_bars.clear()
        self.ax_bars.set_facecolor(PANEL)
        top5 = np.argsort(probs)[-5:]
        colors = [GREEN if c == orig_class else
                  (MAGENTA if c == pred and fooled else '#3b4a5a')
                  for c in top5]
        self.ax_bars.barh(range(5), probs[top5], color=colors, height=0.62)
        self.ax_bars.set_yticks(range(5))
        self.ax_bars.set_yticklabels([short_name(cats[c], 20) for c in top5],
                                     color=FG, fontsize=9)
        self.ax_bars.set_xlim(0, 1)
        self.ax_bars.set_title('Model beliefs (top 5)', color=FG,
                               fontsize=11, pad=8)
        self.ax_bars.tick_params(colors=MUTED, labelsize=8)
        self.ax_bars.grid(True, axis='x', alpha=0.15, color=MUTED)
        for y, c in zip(range(5), top5):
            inside = probs[c] > 0.82
            self.ax_bars.text(probs[c] - 0.02 if inside else probs[c] + 0.02,
                              y, f"{probs[c]:.1%}", va='center',
                              ha='right' if inside else 'left',
                              color=BG if inside else MUTED,
                              fontsize=8.5,
                              fontweight='bold' if inside else 'normal')

        # --- confidence curve
        self.ax_curve.clear()
        self.ax_curve.set_facecolor(PANEL)
        upto = snap['iter']
        history = [s for s in self.snaps if s['iter'] <= upto]
        xs = [s['iter'] for s in history]
        true_conf = [s['probs'][orig_class] for s in history]
        other = np.array([np.delete(s['probs'], orig_class).max()
                          for s in history])
        for lw, a in ((5, 0.15), (2.5, 1.0)):  # soft glow under the lines
            self.ax_curve.plot(xs, true_conf, color=GREEN, lw=lw, alpha=a)
            self.ax_curve.plot(xs, other, color=RED, lw=lw, alpha=a)
        if self.flip_iter is not None and upto >= self.flip_iter:
            self.ax_curve.axvline(self.flip_iter, color=AMBER, lw=1,
                                  ls='--', alpha=0.8)
            self.ax_curve.text(self.flip_iter, 1.02, ' prediction flips',
                               color=AMBER, fontsize=8.5)
        self.ax_curve.set_xlim(0, self.snaps[-1]['iter'])
        self.ax_curve.set_ylim(0, 1.0)
        self.ax_curve.set_title('Confidence during attack', color=FG,
                                fontsize=11, pad=8)
        self.ax_curve.set_xlabel('PGD iteration', color=MUTED, fontsize=9)
        self.ax_curve.tick_params(colors=MUTED, labelsize=8)
        self.ax_curve.grid(True, alpha=0.15, color=MUTED)
        self.ax_curve.legend(
            [f'true: {short_name(cats[orig_class], 16)}', 'best other class'],
            loc='center right', fontsize=8, facecolor=PANEL,
            edgecolor='#30363d', labelcolor=FG)

        # --- HUD
        self.hud.set_text(f"iter {snap['iter']:>3d}")
        self.metrics.set_text(
            f"ε = {self.epsilon} (L∞)   ‖δ‖₂ = {snap['l2']:.2f}")

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[..., :3].copy()


def ease(t):
    return 3 * t ** 2 - 2 * t ** 3


def build_frames(renderer, snaps, fps):
    """Assemble the frame sequence: intro hold, eased attack, outro hold."""
    frames = []
    intro = renderer.render(snaps[0])
    frames.extend([intro] * int(1.5 * fps))

    n_attack_frames = int(6.0 * fps)
    for f in range(n_attack_frames):
        t = ease((f + 1) / n_attack_frames)
        idx = min(int(round(t * (len(snaps) - 1))), len(snaps) - 1)
        frames.append(renderer.render(snaps[idx]))

    outro = renderer.render(snaps[-1], final=True)
    frames.extend([outro] * int(2.5 * fps))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--image', required=True,
                        help='Path or URL of the input image')
    parser.add_argument('--out', default='demo.mp4',
                        help='Output file (.mp4 or .gif)')
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--iters', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=0.0012)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width in pixels (16:9)')
    args = parser.parse_args()

    print('Loading model...')
    model, categories = load_model(torch.device('cpu'))

    print(f'Loading image: {args.image}')
    if args.image.startswith(('http://', 'https://')):
        pil = load_image_from_url(args.image)
    else:
        pil = load_image_from_file(args.image)
    x = build_transform()(pil).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    orig_class = int(probs.argmax())
    print(f'Clean prediction: {categories[orig_class]} '
          f'({probs[orig_class]:.1%})')

    print(f'Recording PGD attack ({args.iters} iterations)...')
    label = torch.tensor([orig_class])
    snaps = record_pgd_attack(model, x, label, epsilon=args.epsilon,
                              alpha=args.alpha, iters=args.iters)
    final_class = int(np.argmax(snaps[-1]['probs']))
    print(f'Final prediction: {categories[final_class]} '
          f"({snaps[-1]['probs'][final_class]:.1%})")

    print('Rendering frames...')
    fps = args.fps if args.out.lower().endswith('.mp4') else min(args.fps, 15)
    renderer = DemoRenderer(snaps, categories, orig_class, args.epsilon,
                            width=args.width)
    frames = build_frames(renderer, snaps, fps)

    print(f'Encoding {len(frames)} frames -> {args.out}')
    import imageio
    if args.out.lower().endswith('.gif'):
        imageio.mimsave(args.out, frames[::2], fps=fps // 2, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=fps, quality=8,
                        macro_block_size=1)
    print('Done.')


if __name__ == '__main__':
    main()
