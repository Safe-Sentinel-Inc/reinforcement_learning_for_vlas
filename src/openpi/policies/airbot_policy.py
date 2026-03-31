import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    action_dim: int = 32
    num_bins: int = 201

    def __call__(self, data: dict) -> dict:

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        image_dict = {}
        image_mask_dict = {}
        image_name = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        for name in image_name:
            if name in data:
                img = _parse_image(data[name])
                # No need for transformation
                # img = img[..., ::-1]  # BGR -> RGB
                image_dict[name] = img
                image_mask_dict[name] = np.True_

        # # debug
        # import uuid
        # from PIL import Image
        # import os
        # debug_dir = "./debug"
        # os.makedirs(debug_dir, exist_ok=True)
        # uid = uuid.uuid4().hex[:8]
        # for name in image_name:
        #     Image.fromarray(image_dict[name]).save(
        #         f"{debug_dir}/{name}_{uid}.jpg",
        #         quality=95
        #     )

        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
        }
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "binned_value" in data:
            # For value function, clamp binned_value to valid range [0, num_bins).
            data["binned_value"] = data["binned_value"] % self.num_bins
            inputs["binned_value"] = np.asarray(data["binned_value"], dtype=np.int32)
        if "stage" in data:
            inputs["stage"] = np.asarray(data["stage"], dtype=np.int32)
        if "advantage" in data:
            inputs["advantage"] = data["advantage"]
        if "intervention" in data:
            inputs["intervention"] = data["intervention"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            return {"actions": np.asarray(data["actions"])}
        elif "binned_value" in data:
            return {"binned_value": np.asarray(data["binned_value"], dtype=np.int32)}
        else:
            raise ValueError("AirbotOutputs expects either 'actions' or 'binned_value' in the input data.")