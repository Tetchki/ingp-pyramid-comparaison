import dataclasses
from typing import Optional, Dict, Any

import gin

import drjit as dr

def _validate_beta_pair(beta_1: float, beta_2: Optional[float]) -> None:
    if not (0.0 < beta_1 < 1.0):
        raise ValueError("`beta_1` must be in (0, 1).")
    if beta_2 is not None and not (0.0 < beta_2 < 1.0):
        raise ValueError("`beta_2` must be in (0, 1) when provided.")

def _validate_positive_int(name: str, value: int, min_allowed: int = 1) -> None:
    if not isinstance(value, int) or value < min_allowed:
        raise ValueError(f"`{name}` must be an integer >= {min_allowed}.")

@gin.configurable
@dataclasses.dataclass
class SceneConfig:
    """
    Configuration for a scene optimization run.
    """
    scene_path: Optional[str] = None
    ref_image_path: Optional[str] = None

    # Paths
    output_path: str = "./output"

    # Rendering backend
    mitsuba_variant: str = "cuda_ad_rgb"

    # Optimization hyperparameters
    lr: float = 1e-3
    iterations: int = 128

    # Optimizer parameters
    beta_1: float = 0.9
    beta_2: Optional[float] = None

    # Rendering parameters
    spp: int = 1024  # Samples per pixel for rendering during optimization
    spp_primal: int = 16  # Samples per pixel for primal rendering
    spp_grad: int = 1    # Samples per pixel for gradient estimation

    rerender_spp: int = 1024  # Samples per pixel for final rerendering

    # Save intermediate result every N iterations. Set to 0 to disable.
    save_interval: int = 0

    def validate(self) -> None:
        if not (self.scene_path or self.ref_image_path):
            raise ValueError(
                "Invalid config: both `scene_path` and `ref_image_path` are None. One must be provided."
            )

        if self.lr <= 0:
            raise ValueError("`lr` must be > 0.")
        if self.iterations <= 0:
            raise ValueError("`iterations` must be > 0.")
        if self.save_interval < 0:
            raise ValueError("`save_interval` must be >= 0.")

        _validate_beta_pair(self.beta_1, self.beta_2)

        _validate_positive_int("spp", self.spp, 1)
        _validate_positive_int("spp_primal", self.spp_primal, 1)
        _validate_positive_int("spp_grad", self.spp_grad, 1)
        _validate_positive_int("rerender_spp", self.rerender_spp, 1)

        if not isinstance(self.mitsuba_variant, str) or not self.mitsuba_variant:
            raise ValueError("`mitsuba_variant` must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def effective_beta_2(self) -> float:
        """Return beta_2, falling back to default derived from beta_1 if not provided."""
        return self.beta_2 if self.beta_2 is not None else (1.0 - dr.square(1.0 - self.beta_1))


@gin.configurable
def make_scene_config(
    scene_path: Optional[str] = None,
    ref_image_path: Optional[str] = None,

    output_path: str = "../results",

    # Optimization hyperparameters
    lr: float = 1e-3,
    iterations: int = 128,

    # Optimizer parameters
    beta_1: float = 0.9,
    beta_2: Optional[float] = None,

    # Rendering parameters
    spp: int = 1024,
    spp_primal: int = 64,
    spp_grad: int = 1,

    # Rerendering parameters
    rerender_spp: int = 1024,

    # IO
    save_interval: int = 0,
) -> SceneConfig:


    cfg = SceneConfig(
        scene_path=scene_path,
        ref_image_path=ref_image_path,
        output_path=output_path,
        lr=lr,
        iterations=iterations,
        beta_1=beta_1,
        beta_2=beta_2,
        spp=spp,
        spp_primal=spp_primal,
        spp_grad=spp_grad,
        rerender_spp=rerender_spp,
        save_interval=save_interval,
    )
    cfg.validate()
    return cfg

def load_scene_config(config_path: str) -> SceneConfig:
    gin.clear_config()
    gin.parse_config_files_and_bindings([config_path], None, skip_unknown=False)
    cfg = SceneConfig()
    cfg.validate()
    return cfg