import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

from dotenv import load_dotenv
import os

load_dotenv()

# 環境変数を取得
project_id = os.getenv("PROJECT_ID")

vertexai.init(project=project_id, location="us-central1")

print(f"Vertex AI initialized with project {project_id}")
generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

prompt = """
This is a photorealistic image of a cookbook placed on a wooden kitchen table.
"""

print("Generating image...")

images = generation_model.generate_images(
    prompt=prompt,
    number_of_images=1,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    language="en",
    # person_generation="allow_all",
)

images[0].save(location="generated_image.png", include_generation_parameters=False)



