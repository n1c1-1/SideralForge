"""Masks between views: several renderings of the same processing (views) combined zone by zone.

- 'Astro: Blend views' lays views over a base view, each through its mask, and shows which view is used where.
- 'Astro: Brightness mask' makes a mask from an image's brightness (a nebula from the starless layer, the stars from
  the stars layer, bright cores from the final image), with thresholds measured on the image itself (percentiles).
- 'Astro: Painted mask' is a mask painted by hand with a brush over an image, in the node (web/astro_paint.js).

Sizes are given as a percentage of the image width, so a mask made at one scale works at any other; a mask of another
size than the image it is used with is resized to it.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import torch
from comfy_api.latest import io

CATEGORY = "astro/masks"
N_VIEWS = 4                                    # views over the base in Astro: Blend views
COLOURS = [(0.45, 0.45, 0.45), (1.0, 0.55, 0.1), (0.15, 0.8, 1.0), (1.0, 0.25, 0.8), (0.4, 1.0, 0.3)]


def _px(percent: float, width: int) -> float:
    return float(percent) / 100.0 * width


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), sigma) if sigma >= 0.3 else a


def _grow(a: np.ndarray, radius: float) -> np.ndarray:
    """Dilation by about 'radius' pixels (alternate square and cross steps: close to a disc), soft values kept."""
    r = int(round(radius))
    if r < 1:
        return a
    sq = np.ones((3, 3), np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = a
    for i in range(r):
        out = cv2.dilate(out, sq if i % 2 else cross)
    return out


def _fit(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def _mask_at(masks: torch.Tensor, b: int) -> np.ndarray:
    m = masks if masks.ndim == 3 else masks[None]
    return m[min(b, m.shape[0] - 1)].float().cpu().numpy()


def _luminance(img: np.ndarray) -> np.ndarray:
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


class AstroBlendViews(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = [io.Image.Input("base", tooltip="The view used wherever no mask takes another one")]
        for k in range(2, N_VIEWS + 2):
            inputs += [io.Image.Input(f"view_{k}", optional=True,
                                      tooltip=f"View {k}: laid over the views before it where mask {k} is white"),
                       io.Mask.Input(f"mask_{k}", optional=True,
                                     tooltip=f"Where view {k} is used (any size: resized to the image)")]
        inputs.append(io.Float.Input("feather", default=0.0, min=0.0, max=5.0, step=0.05,
                                     tooltip="Extra soft edge on every mask, in % of the image width"))
        return io.Schema(
            node_id="AstroBlendViews", display_name="Astro: Blend views", category=CATEGORY,
            description="Combines views of the same image zone by zone: each view is laid over the ones before it "
                        "where its mask is white (soft edges blend). The views must have the same size (render them "
                        "at the same scale). 'coverage' shows which view is used where: grey = the base, then "
                        "orange, cyan, magenta, green for views 2 to 5.",
            inputs=inputs,
            outputs=[io.Image.Output(display_name="image"), io.Image.Output(display_name="coverage")])

    @classmethod
    def execute(cls, base, feather=0.0, **kw) -> io.NodeOutput:
        n, h, w, _ = base.shape
        out = base.float().cpu().numpy().copy()
        weights = [np.ones((n, h, w), np.float32)]           # base, then one per view used
        colours = [COLOURS[0]]
        for k in range(2, N_VIEWS + 2):
            v, m = kw.get(f"view_{k}"), kw.get(f"mask_{k}")
            if v is None and m is None:
                continue
            if v is None or m is None:
                raise ValueError(f"view {k} and mask {k} go together: connect both or neither")
            if tuple(v.shape[1:3]) != (h, w):
                raise ValueError(f"view {k} is {v.shape[2]}x{v.shape[1]}, the base {w}x{h}: render the views at the "
                                 "same scale")
            vv = v.float().cpu().numpy()
            wk = np.empty((n, h, w), np.float32)
            for b in range(n):
                mk = np.clip(_blur(_fit(_mask_at(m, b), h, w), _px(feather, w) / 2), 0, 1)
                out[b] = out[b] * (1 - mk[..., None]) + vv[min(b, len(vv) - 1)] * mk[..., None]
                wk[b] = mk
            for prev in weights:
                prev *= 1 - wk
            weights.append(wk)
            colours.append(COLOURS[(k - 1) % len(COLOURS)])
        cov = np.zeros_like(out)
        for wk, c in zip(weights, colours):
            cov += wk[..., None] * np.array(c, np.float32)
        lum = _luminance(out)[..., None]
        cov = np.clip(0.45 * lum + 0.55 * cov, 0, 1)
        return io.NodeOutput(torch.from_numpy(np.clip(out, 0, 1)), torch.from_numpy(cov))


class AstroBrightnessMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroBrightnessMask", display_name="Astro: Brightness mask", category=CATEGORY,
            description="A mask from an image's brightness, smoothed into zones: 0 below 'low', 1 above 'high', a "
                        "soft ramp between (low above high selects the dark parts). Thresholds in percentiles of the "
                        "image (they adapt to each image) or in levels (% of white). Examples: the nebula = the "
                        "starless layer from percentile 60 to 90; the stars = the stars layer, levels 3 to 15 %, "
                        "grown; bright cores = the final image, percentiles 99.5 to 99.95.",
            inputs=[io.Image.Input("image", tooltip="The image whose brightness makes the mask"),
                    io.Combo.Input("unit", options=["percentile", "level"], default="percentile",
                                   tooltip="percentile: thresholds measured on this image (0-100); level: brightness "
                                           "in % of white (0-100)"),
                    io.Float.Input("low", default=60.0, min=0.0, max=100.0, step=0.1,
                                   tooltip="Brightness where the mask starts (0 below)"),
                    io.Float.Input("high", default=90.0, min=0.0, max=100.0, step=0.1,
                                   tooltip="Brightness where the mask is full (1 above)"),
                    io.Float.Input("smooth", default=0.4, min=0.0, max=10.0, step=0.05,
                                   tooltip="Blur of the brightness before the thresholds (zones rather than pixels), "
                                           "in % of the image width"),
                    io.Float.Input("grow", default=0.0, min=0.0, max=10.0, step=0.05,
                                   tooltip="Widen the mask, in % of the image width"),
                    io.Float.Input("feather", default=0.2, min=0.0, max=10.0, step=0.05,
                                   tooltip="Soft edge, in % of the image width"),
                    io.Boolean.Input("invert", default=False, tooltip="Swap the mask (1 - mask)")],
            outputs=[io.Mask.Output(display_name="mask"), io.String.Output(display_name="thresholds")])

    @classmethod
    def execute(cls, image, unit, low, high, smooth, grow, feather, invert) -> io.NodeOutput:
        n, h, w, _ = image.shape
        masks, used = [], []
        for b in range(n):
            lum = _blur(_luminance(image[b].float().cpu().numpy()), _px(smooth, w) / 2)
            if unit == "percentile":
                lo, hi = (float(v) for v in np.percentile(lum[::2, ::2], [low, high]))
            else:
                lo, hi = float(low) / 100.0, float(high) / 100.0
            span = hi - lo
            m = np.clip((lum - lo) / span, 0, 1) if abs(span) > 1e-6 else (lum >= lo).astype(np.float32)
            m = (m * m * (3 - 2 * m)).astype(np.float32)          # smooth ramp
            m = _blur(_grow(m, _px(grow, w)), _px(feather, w) / 2)
            masks.append(1 - m if invert else m)
            used.append(f"{lo:.4f} -> {hi:.4f}")
        return io.NodeOutput(torch.from_numpy(np.stack(masks).astype(np.float32)),
                             "levels " + "; ".join(used))


def _decode_painted(data: str) -> np.ndarray | None:
    """The painted mask kept by the node: a PNG (data URL) whose alpha channel is the mask."""
    if not data or "," not in data:
        return None
    raw = np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8)
    a = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., 3]
    elif a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32) / 255.0


class AstroPaintedMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AstroPaintedMask", display_name="Astro: Painted mask", category=CATEGORY, is_output_node=True,
            description="A mask painted by hand over an image: run the graph once to show the image in the node, "
                        "then paint (left button) or erase (right button or Erase). The mask is kept in the node "
                        "(saved with the graph) and follows the image at any scale.",
            inputs=[io.Image.Input("image", tooltip="The image to paint over (a view of the same processing)"),
                    io.String.Input("painted", default="", optional=True,
                                    tooltip="The painted mask, kept by the node (a PNG)"),
                    io.Float.Input("feather", default=0.0, min=0.0, max=10.0, step=0.05,
                                   tooltip="Extra soft edge, in % of the image width"),
                    io.Boolean.Input("invert", default=False, tooltip="Swap the mask (1 - mask)")],
            outputs=[io.Mask.Output(display_name="mask")])

    @classmethod
    def execute(cls, image, feather, invert, painted="") -> io.NodeOutput:
        import uuid

        import folder_paths
        n, h, w, _ = image.shape
        # the image to paint over, at most 1600 pixels wide, for the node's canvas
        a = np.clip(image[0].float().cpu().numpy(), 0, 1)
        if w > 1600:
            a = cv2.resize(a, (1600, round(h * 1600 / w)), interpolation=cv2.INTER_AREA)
        name = f"astro_paint_{uuid.uuid4().hex[:12]}.jpg"
        cv2.imwrite(f"{folder_paths.get_temp_directory()}/{name}", (a[..., ::-1] * 255 + 0.5).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        m = _decode_painted(painted)
        m = np.zeros((h, w), np.float32) if m is None else _fit(m, h, w)
        m = np.clip(_blur(m, _px(feather, w) / 2), 0, 1)
        if invert:
            m = 1 - m
        ui = {"astro_paint": [{"filename": name, "subfolder": "", "type": "temp", "width": w, "height": h}]}
        return io.NodeOutput(torch.from_numpy(np.repeat(m[None], n, 0).astype(np.float32)), ui=ui)


MASK_NODES = [AstroBlendViews, AstroBrightnessMask, AstroPaintedMask]
