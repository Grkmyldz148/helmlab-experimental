#!/usr/bin/env python3
"""Screenshot blog HTML visuals as JPEG at 100% quality using Playwright."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent / "blog_output"
VISUALS = [
    ("hero.html", 1200, 630),
    ("gradient_comparison.html", 1200, 800),
    ("benchmark.html", 1200, 600),
    ("blue_white.html", 1200, 600),
    ("stats.html", 1200, 500),
]

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for name, w, h in VISUALS:
            html_path = OUT / name
            if not html_path.exists():
                print(f"SKIP {name} (not found)")
                continue
            page = await browser.new_page(viewport={"width": w, "height": h})
            await page.goto(f"file://{html_path}")
            await page.wait_for_timeout(500)
            jpg_path = OUT / name.replace(".html", ".jpg")
            await page.screenshot(path=str(jpg_path), type="jpeg", quality=100, full_page=False)
            print(f"OK {jpg_path.name} ({jpg_path.stat().st_size // 1024} KB)")
            await page.close()
        await browser.close()

asyncio.run(main())
