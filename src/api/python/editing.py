import vertexai

from vertexai.preview.vision_models import (
    Image,
    ImageGenerationModel,
    MaskReferenceImage,
    RawReferenceImage,
    ControlReferenceImage
)
import io

from PIL import Image as PIL_Image

import cv2

from dotenv import load_dotenv
import os

load_dotenv()

# 環境変数を取得
project_id = os.getenv("PROJECT_ID")

print(f"Vertex AI initialized with project {project_id}")
vertexai.init(project=project_id, location="us-central1")





# Gets the image bytes from a PIL Image object.
def get_bytes_from_pil(image: PIL_Image) -> bytes:
    byte_io_png = io.BytesIO()
    image.save(byte_io_png, "PNG")
    return byte_io_png.getvalue()


# Pads an image for outpainting.
def pad_to_target_size(
    source_image,
    target_size=(1536, 1536),
    mode="RGB",
    vertical_offset_ratio=0,
    horizontal_offset_ratio=0,
    fill_val=255,
):
    orig_image_size_w, orig_image_size_h = source_image.size
    target_size_w, target_size_h = target_size

    insert_pt_x = (target_size_w - orig_image_size_w) // 2 + int(
        horizontal_offset_ratio * target_size_w
    )
    insert_pt_y = (target_size_h - orig_image_size_h) // 2 + int(
        vertical_offset_ratio * target_size_h
    )
    insert_pt_x = min(insert_pt_x, target_size_w - orig_image_size_w)
    insert_pt_y = min(insert_pt_y, target_size_h - orig_image_size_h)

    if mode == "RGB":
        source_image_padded = PIL_Image.new(
            mode, target_size, color=(fill_val, fill_val, fill_val)
        )
    elif mode == "L":
        source_image_padded = PIL_Image.new(mode, target_size, color=(fill_val))
    else:
        raise ValueError("source image mode must be RGB or L.")

    source_image_padded.paste(source_image, (insert_pt_x, insert_pt_y))
    return source_image_padded


# Pads and resizes image and mask to the same target size.
def pad_image_and_mask(
    image_vertex: Image,
    mask_vertex: Image,
    target_size,
    vertical_offset_ratio,
    horizontal_offset_ratio,
):
    image_vertex.thumbnail(target_size)
    mask_vertex.thumbnail(target_size)

    image_vertex = pad_to_target_size(
        image_vertex,
        target_size=target_size,
        mode="RGB",
        vertical_offset_ratio=vertical_offset_ratio,
        horizontal_offset_ratio=horizontal_offset_ratio,
        fill_val=0,
    )
    mask_vertex = pad_to_target_size(
        mask_vertex,
        target_size=target_size,
        mode="L",
        vertical_offset_ratio=vertical_offset_ratio,
        horizontal_offset_ratio=horizontal_offset_ratio,
        fill_val=255,
    )
    return image_vertex, mask_vertex



     

generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

edit_model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")

image_prompt = """
a small wooden bowl with grapes and apples on a marble kitchen counter, light brown cabinets blurred in the background
"""
print("Generating image...")
generated_image = generation_model.generate_images(
    prompt=image_prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="dont_allow",
)

edit_prompt = "a small white ceramic bowl with lemons and limes"
raw_ref_image = RawReferenceImage(image=generated_image[0], reference_id=0)
mask_ref_image = MaskReferenceImage(
    reference_id=1, image=None, mask_mode="foreground", dilation=0.1
)
edited_image = edit_model.edit_image(
    prompt=edit_prompt,
    edit_mode="inpainting-insert",
    reference_images=[raw_ref_image, mask_ref_image],
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

generated_image[0].save(location="generated_image0.png", include_generation_parameters=False)
edited_image[0].save(location="edited_image0.png", include_generation_parameters=False)



image_prompt = """
a french bulldog sitting in a living room on a couch with green throw pillows and a throw blanket,
a circular mirror with a slim black border is on the wall above the couch
"""
generated_image = generation_model.generate_images(
    prompt=image_prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="dont_allow",
)

edit_prompt = "a corgi sitting on a couch"
raw_ref_image = RawReferenceImage(image=generated_image[0], reference_id=0)
mask_ref_image = MaskReferenceImage(
    reference_id=1,
    image=None,
    mask_mode="semantic",
    segmentation_classes=[8],
    dilation=0.1,
)
edited_image = edit_model.edit_image(
    prompt=edit_prompt,
    edit_mode="inpainting-insert",
    reference_images=[raw_ref_image, mask_ref_image],
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)


generated_image[0].save(location="generated_image1.png", include_generation_parameters=False)
edited_image[0].save(location="edited_image1.png", include_generation_parameters=False)


raw_ref_image = RawReferenceImage(image=edited_image[0], reference_id=0)
mask_ref_image = MaskReferenceImage(
    reference_id=1, image=None, mask_mode="semantic", segmentation_classes=[85]
)
remove_image = edit_model.edit_image(
    prompt="",
    edit_mode="inpainting-remove",
    reference_images=[raw_ref_image, mask_ref_image],
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

remove_image[0].save(location="remove_image0.png", include_generation_parameters=False)




product_image = Image(
    gcs_uri="gs://cloud-samples-data/generative-ai/image/suitcase.png"
)
raw_ref_image = RawReferenceImage(image=product_image, reference_id=0)
mask_ref_image = MaskReferenceImage(reference_id=1, image=None, mask_mode="background")

prompt = "a light blue suitcase in front of a window in an airport, lots of bright, natural lighting coming in from the windows, planes taking off in the distance"
edited_image = edit_model.edit_image(
    prompt=prompt,
    edit_mode="background-swap",
    reference_images=[raw_ref_image, mask_ref_image],
    seed=1,
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

edited_image[0].save(location="edited_image2.png", include_generation_parameters=False)



initial_image = Image(
    gcs_uri="gs://cloud-samples-data/generative-ai/image/living-room.png"
)
mask = PIL_Image.new("L", initial_image._pil_image.size, 0)

target_size_w = int(2500 * eval("3/4"))
target_size = (target_size_w, 2500)
image_pil_outpaint, mask_pil_outpaint = pad_image_and_mask(
    initial_image._pil_image,
    mask,
    target_size,
    0,
    0,
)

raw_ref_image = RawReferenceImage(
    image=get_bytes_from_pil(image_pil_outpaint), reference_id=0
)
mask_ref_image = MaskReferenceImage(
    reference_id=1,
    image=get_bytes_from_pil(mask_pil_outpaint),
    mask_mode="user_provided",
    dilation=0.03,
)
prompt = "a chandelier hanging from the ceiling"
edited_image = edit_model.edit_image(
    prompt=prompt,
    edit_mode="outpainting",
    reference_images=[raw_ref_image, mask_ref_image],
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

edited_image[0].save(location="edited_image3.png", include_generation_parameters=False)



generation_prompt = """
a simple accent chair in a neutral color
"""
generated_image = generation_model.generate_images(
    prompt=generation_prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="dont_allow",
)

generated_image[0].save("chair.png")
img = cv2.imread("chair.png")

# Setting parameter values
t_lower = 100  # Lower Threshold
t_upper = 150  # Upper threshold

# Applying the Canny Edge filter
edge = cv2.Canny(img, t_lower, t_upper)
cv2.imwrite("chair_edge.png", edge)

control_image = ControlReferenceImage(
    reference_id=1, image=Image.load_from_file("chair_edge.png"), control_type="canny"
)

edit_prompt = "A photorealistic image along the lines of a navy suede accent chair in a living room, near big windows"

control_image = edit_model._generate_images(
    prompt=edit_prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    reference_images=[control_image],
    safety_filter_level="block_some",
    person_generation="allow_adult",
)


control_image[0].save(location="edited_image4.png", include_generation_parameters=False)
