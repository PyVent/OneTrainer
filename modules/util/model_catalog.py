from modules.util.enum.ModelType import ModelType

# UI order and labels shared by training and model conversion. Legacy SD base/depth
# variants remain loadable through saved configs but are not separate UI entries.
MODEL_TYPE_CHOICES: tuple[tuple[str, ModelType], ...] = (
    ("SD1.5", ModelType.STABLE_DIFFUSION_15),
    ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
    ("SD2.0", ModelType.STABLE_DIFFUSION_20),
    ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
    ("SD2.1", ModelType.STABLE_DIFFUSION_21),
    ("SD3", ModelType.STABLE_DIFFUSION_3),
    ("SD3.5", ModelType.STABLE_DIFFUSION_35),
    ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
    ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
    ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
    ("Stable Cascade", ModelType.STABLE_CASCADE_1),
    ("PixArt Alpha", ModelType.PIXART_ALPHA),
    ("PixArt Sigma", ModelType.PIXART_SIGMA),
    ("Flux Dev.1", ModelType.FLUX_DEV_1),
    ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
    ("Flux 2 [Dev, Klein]", ModelType.FLUX_2),
    ("Sana", ModelType.SANA),
    ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
    ("HiDream Full", ModelType.HI_DREAM_FULL),
    ("Chroma1", ModelType.CHROMA_1),
    ("QwenImage", ModelType.QWEN),
    ("Anima", ModelType.ANIMA),
    ("Anima (qwen 2.1 vae)", ModelType.ANIMA_QWEN21_VAE),
    ("Krea 2", ModelType.KREA_2),
    ("Z-Image", ModelType.Z_IMAGE),
    ("Ernie Image", ModelType.ERNIE),
    ("Ideogram 4", ModelType.IDEOGRAM_4),
)
