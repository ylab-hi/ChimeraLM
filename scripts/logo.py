# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pillow",
#     "typer",
# ]
# ///
from PIL import Image
from typer import Typer
from pathlib import Path

app = Typer()

@app.command()
def image_to_terminal(
    image_path: Path,
    max_width: int = None,
    white_threshold: int = 240
):
    """
    Convert image using half-blocks (▀ ▄) - 2 pixels per character.
    Maintains full detail while using half the height!
    """
    img = Image.open(image_path)
    img = img.convert("RGB")

    if max_width and img.width > max_width:
        aspect_ratio = img.height / img.width
        new_height = int(max_width * aspect_ratio)
        img = img.resize((max_width, new_height))

    width, height = img.size

    for y in range(0, height, 2):  # Process 2 rows at a time
        for x in range(width):
            # Top pixel
            r1, g1, b1 = img.getpixel((x, y))
            top_is_white = r1 > white_threshold and g1 > white_threshold and b1 > white_threshold

            # Bottom pixel (if exists)
            if y + 1 < height:
                r2, g2, b2 = img.getpixel((x, y + 1))
                bottom_is_white = r2 > white_threshold and g2 > white_threshold and b2 > white_threshold
            else:
                bottom_is_white = True
                r2, g2, b2 = 255, 255, 255

            # Choose character and colors
            if top_is_white and bottom_is_white:
                print(" ", end="")
            elif top_is_white and not bottom_is_white:
                # Bottom half: ▄
                print(f"\033[38;2;{r2};{g2};{b2}m▄\033[0m", end="")
            elif not top_is_white and bottom_is_white:
                # Top half: ▀
                print(f"\033[38;2;{r1};{g1};{b1}m▀\033[0m", end="")
            else:
                # Both colored: use top color for ▀, background for bottom
                print(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▀\033[0m", end="")
        print()

if __name__ == "__main__":
    app()
