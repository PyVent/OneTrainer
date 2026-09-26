from PIL import Image, ImageOps


def has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def composite_on_white(image: Image.Image) -> Image.Image:
    """Flatten transparency for output formats such as JPEG, which cannot store alpha."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def load_image(path: str, convert_mode: str = 'RGB') -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if convert_mode:
        image = image.convert(convert_mode)
    return image
