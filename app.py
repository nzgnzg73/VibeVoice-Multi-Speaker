import os
import sys
import spaces
import torch
import numpy as np
import soundfile as sf
import librosa
import logging
import gradio as gr
import tempfile
import re
from typing import Dict, Optional

# --- 1. Setup Environment ---

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("VibeVoiceGradio")

# Mock ComfyUI's folder_paths module
class MockFolderPaths:
    def get_folder_paths(self, folder_name):
        if folder_name == "checkpoints":
            models_dir = os.path.join(project_root, "models")
            os.makedirs(models_dir, exist_ok=True)
            return [models_dir]
        return []

sys.modules['folder_paths'] = MockFolderPaths()

# Import BOTH node classes
from nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
from nodes.multi_speaker_node import VibeVoiceMultipleSpeakersNode

# --- 2. Load Models and Share Weights ---

logger.info("Initializing VibeVoice nodes...")
# Instantiate both node types.
single_speaker_node = VibeVoiceSingleSpeakerNode()
multi_speaker_node = VibeVoiceMultipleSpeakersNode()

try:
    logger.info("Loading VibeVoice-Large model once. This may take a while on the first run...")
    # Load the model into one node first.
    multi_speaker_node.load_model(
        model_name='VibeVoice-Large',
        model_path='aoi-ot/VibeVoice-Large',
        attention_type='auto'
    )
    
    logger.info("Sharing loaded model weights between node instances...")
    single_speaker_node.model = multi_speaker_node.model
    single_speaker_node.processor = multi_speaker_node.processor
    single_speaker_node.current_model_path = multi_speaker_node.current_model_path
    single_speaker_node.current_attention_type = multi_speaker_node.current_attention_type
    
    logger.info("VibeVoice-Large model loaded and shared successfully!")

except Exception as e:
    logger.error(f"Failed to load the model: {e}", exc_info=True)
    logger.error("Please ensure you have an internet connection for the first run and sufficient VRAM.")
    sys.exit(1)


# --- 3. Helper Functions ---

def load_audio_for_node(file_path: Optional[str]) -> Optional[Dict]:
    """Loads an audio file and formats it for the node."""
    if file_path is None:
        return None
    try:
        waveform, sr = librosa.load(file_path, sr=24000, mono=True)
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
        return {"waveform": waveform_tensor, "sample_rate": 24000}
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        return None

def save_audio_to_tempfile(audio_dict: Dict) -> Optional[str]:
    """Saves the node's audio output to a temporary WAV file for Gradio."""
    if not audio_dict or "waveform" not in audio_dict:
        return None
    
    waveform_np = audio_dict["waveform"].squeeze().cpu().numpy()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, waveform_np, audio_dict["sample_rate"])
        return tmpfile.name

def detect_speaker_count(text: str) -> int:
    """Analyzes text to count the number of unique speakers."""
    speaker_tags = re.findall(r'\[(\d+)\]\s*:', text)
    if not speaker_tags:
        # No tags found, treat as a single speaker monologue.
        return 1
    unique_speakers = set(int(tag) for tag in speaker_tags)
    return len(unique_speakers)

# --- 4. Gradio Core Logic ---

@spaces.GPU
def generate_speech_gradio(
    text: str,
    speaker1_audio_path: Optional[str],
    speaker2_audio_path: Optional[str],
    speaker3_audio_path: Optional[str],
    speaker4_audio_path: Optional[str],
    seed: int,
    diffusion_steps: int,
    cfg_scale: float,
    use_sampling: bool,
    temperature: float,
    top_p: float,
    max_words_per_chunk: int,
    progress=gr.Progress(track_tqdm=True)
):
    """The main function that Gradio will call, now with dynamic node switching."""
    if not text or not text.strip():
        raise gr.Error("Please provide some text to generate.")

    progress(0, desc="Analyzing text and loading voices...")
    
    speaker_count = detect_speaker_count(text)
    
    # Load voices
    speaker1_voice = load_audio_for_node(speaker1_audio_path)
    speaker2_voice = load_audio_for_node(speaker2_audio_path)
    speaker3_voice = load_audio_for_node(speaker3_audio_path)
    speaker4_voice = load_audio_for_node(speaker4_audio_path)

    progress(0.2, desc="Generating speech... (this can take a moment)")

    try:
        if speaker_count <= 1:
            logger.info(f"Detected single speaker. Using VibeVoiceSingleSpeakerNode.")
            # Prepare text for single speaker node (remove tags like [1]:)
            processed_text = re.sub(r'\[1\]\s*:', '', text).strip()
            
            audio_output_tuple = single_speaker_node.generate_speech(
                text=processed_text,
                model='VibeVoice-Large',
                attention_type='auto',
                free_memory_after_generate=False,
                diffusion_steps=int(diffusion_steps),
                seed=int(seed),
                cfg_scale=cfg_scale,
                use_sampling=use_sampling,
                voice_to_clone=speaker1_voice, # Use speaker 1's voice for cloning
                temperature=temperature,
                top_p=top_p,
                max_words_per_chunk=int(max_words_per_chunk)
            )
        else:
            logger.info(f"Detected {speaker_count} speakers. Using VibeVoiceMultipleSpeakersNode.")
            audio_output_tuple = multi_speaker_node.generate_speech(
                text=text,
                model='VibeVoice-Large',
                attention_type='auto',
                free_memory_after_generate=False,
                diffusion_steps=int(diffusion_steps),
                seed=int(seed),
                cfg_scale=cfg_scale,
                use_sampling=use_sampling,
                speaker1_voice=speaker1_voice,
                speaker2_voice=speaker2_voice,
                speaker3_voice=speaker3_voice,
                speaker4_voice=speaker4_voice,
                temperature=temperature,
                top_p=top_p
            )
    except Exception as e:
        logger.error(f"Error during speech generation: {e}", exc_info=True)
        raise gr.Error(f"An error occurred during generation: {e}")
    
    progress(0.9, desc="Saving audio file...")
    output_audio_path = save_audio_to_tempfile(audio_output_tuple[0])
    
    if output_audio_path is None:
        raise gr.Error("Failed to process the generated audio.")

    return output_audio_path

# --- 5. Gradio UI Layout ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# VibeVoice Text-to-Speech Demo\n"
        "Generate single or multi-speaker audio. For single-speaker monologues, the system automatically uses a specialized node with text chunking."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text Input",
                placeholder=(
                    "Enter plain text for a single speaker, or use tags like [1]:, [2]: for multiple speakers.\n\n"
                    "[1]: Hello, I'm the first speaker.\n"
                    "[2]: Hi there, I'm the second! How are you?"
                ),
                lines=8,
                max_lines=20
            )
            with gr.Accordion("Upload Speaker Voices (Optional)", open=True):
                gr.Markdown("Upload a short audio clip (3-30 seconds, clear audio) for each speaker you want to clone.")
                with gr.Row():
                    speaker1_audio = gr.Audio(label="Speaker 1 Voice", type="filepath")
                    speaker2_audio = gr.Audio(label="Speaker 2 Voice", type="filepath")
                with gr.Row():
                    speaker3_audio = gr.Audio(label="Speaker 3 Voice", type="filepath")
                    speaker4_audio = gr.Audio(label="Speaker 4 Voice", type="filepath")
            
            with gr.Accordion("Advanced Options", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=2**32-1, step=1, value=42, interactive=True)
                diffusion_steps = gr.Slider(label="Diffusion Steps", minimum=5, maximum=100, step=1, value=20, interactive=True, info="More steps = better quality, but slower.")
                cfg_scale = gr.Slider(label="CFG Scale", minimum=0.5, maximum=3.5, step=0.05, value=1.3, interactive=True, info="Guidance scale.")
                use_sampling = gr.Checkbox(label="Use Sampling", value=False, interactive=True, info="Enable for more varied, less deterministic output.")
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.95, interactive=True, info="Only used when sampling is enabled.")
                top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, step=0.05, value=0.95, interactive=True, info="Only used when sampling is enabled.")
                max_words_per_chunk = gr.Slider(label="Max Words Per Chunk", minimum=100, maximum=500, step=10, value=250, interactive=True, info="For long single-speaker text. Splits text to avoid errors.")
        
        with gr.Column(scale=1):
            generate_button = gr.Button("Generate Speech", variant="primary")
            audio_output = gr.Audio(label="Generated Speech", type="filepath", interactive=False)

    inputs = [
        text_input,
        speaker1_audio, speaker2_audio, speaker3_audio, speaker4_audio,
        seed, diffusion_steps, cfg_scale, use_sampling, temperature, top_p, max_words_per_chunk
    ]
    
    generate_button.click(
        fn=generate_speech_gradio,
        inputs=inputs,
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()