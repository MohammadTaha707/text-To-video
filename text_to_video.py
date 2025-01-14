from diffusers import StableDiffusionPipeline
import torch

# Load the pretrained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")  # Move the model to GPU

# # Test the pipeline with a simple text prompt
# prompt = "A scenic sunset over the ocean with waves"
# image = pipe(prompt).images[0]
# image.show()


import numpy as np
import cv2


def generate_video_from_text(text: str) -> str:
    # Frame generation settings
    fps = 30  # Frames per second
    num_frames = 300  # 10 sec of video at 30 FPS  10*30
    text_prompt = "cars falling from heavens"
    frame_width, frame_height = 512, 512

    # Set up the video writer
    output_video = "generated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Generate each frame and write to video
    for frame_idx in range(num_frames):
        frame_prompt = f"{text_prompt}, frame {frame_idx}"
        image = pipe(frame_prompt).images[0]  # Generate the frame
        frame = np.array(image)  # Convert PIL Image to NumPy array
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write to video

    video_writer.release()
    return output_video
# print(f"Video saved as {output_video}")


if __name__ == "__main__":
    text_input = "A dog playing in the park"
    output_video = generate_video_from_text(text_input)
    print(f"Video saved at {output_video}")
