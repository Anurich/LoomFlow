"""33_decisions_images.py — System One decisions over IMAGES, plus Box.

Jev reads images (1.13+) and, from Jev 1.14, answers a fourth
question type: ``Box`` — *where* is something in an image. Put a
``loomflow.Image`` anywhere in the state, or inside a question's
instructions or criteria (an example photo per Choice option)::

    photo = Image.from_path("listing.jpg")
    d = await JevModel("jev-preview").decide(
        {"photo": photo},
        questions={
            "present": Noul("Is there a red square in `photo`?"),
            "where":   Box("the red square in `photo`"),
        },
    )
    d.boxes["where"].to_pixels(width, height)   # PIL crop tuple

loomflow adds the ``TypeSafe-Danger-Zone-Images`` opt-in header,
uploads each image once (multipart), rejects GIF/WebP before sending,
and fails fast on Box against a pinned pre-1.14 model.

The scene is generated in pure Python (no Pillow): a 1200x800 PNG
with a red square at KNOWN pixel coordinates, so the Box answer can be
checked against ground truth.

* With ``TYPESAFE_API_KEY`` set → runs on real Jev (``jev-preview``).
* Without → ``ScriptedDecisions`` stands in, so it runs offline.

Run with::

    python examples/33_decisions_images.py
"""

from __future__ import annotations

import asyncio
import os
import struct
import zlib
from typing import Any

from loomflow.decisions import (
    Box,
    BoxDecision,
    Choice,
    ChoiceDecision,
    DecisionModel,
    Image,
    JevModel,
    Noul,
    NoulDecision,
    Score,
    ScoreDecision,
    ScriptedDecisions,
)

# ---------------------------------------------------------------------------
# A scene with known ground truth
# ---------------------------------------------------------------------------

W, H = 1200, 800  # within Jev's 280-1568 px per side
SQUARE = (820, 120, 940, 240)  # left, top, right, bottom (px)
WHITE, RED, BLUE = b"\xff\xff\xff", b"\xd0\x20\x20", b"\x20\x40\xd0"


def _png(width: int, height: int, rows: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def render(x0: int = 0, y0: int = 0, w: int = W, h: int = H) -> Image:
    """The scene, or any window of it (a crop), as a PNG Image."""
    left, top, right, bottom = SQUARE
    rows = []
    for y in range(y0, y0 + h):
        if top <= y < bottom:
            a = min(max(left - x0, 0), w)
            z = min(max(right - x0, 0), w)
            row = WHITE * a + RED * (z - a) + WHITE * (w - z)
        else:
            row = WHITE * w
        rows.append(b"\x00" + row)  # filter byte 0 per scanline
    return Image.from_bytes(_png(w, h, b"".join(rows)))


def swatch(rgb: bytes) -> Image:
    """A solid 280x280 example image for a Choice option."""
    return Image.from_bytes(_png(280, 280, (b"\x00" + rgb * 280) * 280))


# ---------------------------------------------------------------------------
# Decider: real Jev when a key is present, scripted otherwise
# ---------------------------------------------------------------------------


def make_decider() -> tuple[DecisionModel, bool]:
    if os.environ.get("TYPESAFE_API_KEY"):
        return JevModel("jev-preview"), True  # Box needs Jev 1.14+
    # Offline: the answers a well-behaved Jev returns for this scene.
    return (
        ScriptedDecisions(
            answers={
                "present": NoulDecision(noul=0.98),
                "where": BoxDecision(
                    top=0.15, left=0.6833, bottom=0.30, right=0.7833,
                    confidence=0.62,
                ),
                "color": ChoiceDecision(
                    choice="red",
                    probabilities={"red": 0.95, "blue": 0.03, "other": 0.02},
                    confidence=0.95,
                ),
                "coverage": ScoreDecision(
                    score=1.1,
                    legend=["none", "a small part", "about half", "most"],
                    probabilities=[0.05, 0.8, 0.15, 0.0],
                    confidence=0.8,
                ),
                "quadrant": ChoiceDecision(
                    choice="top_right",
                    probabilities={"top_left": 0.01, "top_right": 0.97,
                                   "bottom_left": 0.01, "bottom_right": 0.01},
                    confidence=0.97,
                ),
                "where_in_crop": BoxDecision(
                    top=0.30, left=0.3667, bottom=0.60, right=0.5667,
                    confidence=0.91,
                ),
            }
        ),
        False,
    )


def fmt_px(box: BoxDecision) -> str:
    left, top, right, bottom = box.to_pixels(W, H)
    return f"({left}, {top}) -> ({right}, {bottom}) px"


# ---------------------------------------------------------------------------
# 1) One call: presence, location, example-image Choice, Score
# ---------------------------------------------------------------------------


async def demo_inspect(decider: DecisionModel) -> None:
    photo = render()
    d = await decider.decide(
        {"photo": photo},
        questions={
            # Box always returns a box — pair it with a presence check.
            "present": Noul("Is there a red square in `photo`?"),
            "where": Box("the red square in `photo`"),
            # Options described by example images (+ an undescribed one).
            "color": Choice(
                "What color is the square in `photo`?",
                criteria={
                    "red": {"description": "red", "examples": [swatch(RED)]},
                    "blue": {"description": "blue", "examples": [swatch(BLUE)]},
                    "other": None,
                },
            ),
            "coverage": Score(
                "How much of `photo` does the square cover?",
                criteria=["none", "a small part", "about half", "most"],
            ),
        },
    )
    present = d.nouls["present"].noul
    where = d.boxes["where"]
    print("1) Inspect one photo (4 questions, 1 call)")
    print(f"   present?  P(yes) = {present:.2f}")
    if present > 0.8:
        print(f"   where     {fmt_px(where)}  confidence={where.confidence}")
        print(f"   truth     ({SQUARE[0]}, {SQUARE[1]}) -> ({SQUARE[2]}, {SQUARE[3]}) px")
        if where.confidence is not None and where.confidence < 0.7:
            # TypeSafe's advice for a shaky box: pad it before cropping.
            print(f"   padded    {fmt_px(where.padded(0.02))}  (low confidence)")
    print(f"   color     {d.choices['color'].choice} "
          f"(confidence {d.choices['color'].confidence:.2f})")
    print(f"   coverage  {d.scores['coverage'].score:.2f} / 3 "
          f"→ normalized {d.scores['coverage'].normalized:.2f}")
    print(f"   usage     {d.usage.input_tokens} input tokens, "
          f"${d.usage.cost_usd:.8f}; model={d.model}")
    print()


# ---------------------------------------------------------------------------
# 2) Coarse-to-fine: pick a quadrant, Box the crop, map back
# ---------------------------------------------------------------------------

QUADRANTS: dict[str, tuple[int, int]] = {
    "top_left": (0, 0), "top_right": (W // 2, 0),
    "bottom_left": (0, H // 2), "bottom_right": (W // 2, H // 2),
}


async def demo_coarse_to_fine(decider: DecisionModel) -> None:
    tiles = {name: render(x, y, W // 2, H // 2) for name, (x, y) in QUADRANTS.items()}
    coarse = await decider.decide(
        "Four tiles of one image.",
        questions={
            "quadrant": Choice(
                "Which tile contains the red square?",
                criteria={name: tile for name, tile in tiles.items()},
            )
        },
    )
    name = coarse.choices["quadrant"].choice
    x0, y0 = QUADRANTS[name]
    fine = await decider.decide(
        {"crop": tiles[name]},
        questions={"where_in_crop": Box("the red square in `crop`")},
    )
    crop = BoxDecision(
        top=y0 / H, left=x0 / W,
        bottom=(y0 + H // 2) / H, right=(x0 + W // 2) / W,
    )
    full = fine.boxes["where_in_crop"].to_full_image(crop)
    print("2) Coarse-to-fine (small targets in big images)")
    print(f"   tile      {name}")
    print(f"   box       {fmt_px(full)}  (mapped back from the crop)")
    print()


async def main() -> None:
    decider, live = make_decider()
    print(f"Backend: {'Jev (jev-preview, live)' if live else 'ScriptedDecisions (offline)'}\n")
    try:
        await demo_inspect(decider)
        await demo_coarse_to_fine(decider)
    finally:
        closer: Any = getattr(decider, "aclose", None)
        if closer is not None:
            await closer()
    print("No Jev access yet? LLMDecisionModel(<a vision model>) answers the")
    print("same questions: images ride the prompt as <image N> placeholders.")


if __name__ == "__main__":
    asyncio.run(main())
