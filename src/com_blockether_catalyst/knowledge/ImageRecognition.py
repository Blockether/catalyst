"""
Image recognition module using MiniCPM-V for visual understanding.

This module provides image recognition capabilities for knowledge extraction,
including OCR, visual question answering, and image content understanding.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ImageRecognitionSettings(BaseModel):
    """Settings for image recognition."""

    model_name: str = Field(
        default="openbmb/MiniCPM-V",
        description="The MiniCPM model to use",
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        description="Device to run the model on",
    )
    max_new_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for generation",
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p (nucleus) sampling parameter",
    )
    num_beams: int = Field(
        default=1,
        description="Number of beams for beam search",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether to trust remote code when loading model",
    )
    huggingface_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for accessing gated models. If not provided, will try HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variables",
    )


class ImageAnalysisResult(BaseModel):
    """Result of image analysis."""

    text_content: str = Field(
        description="Extracted text content from the image",
    )
    description: str = Field(
        description="Natural language description of the image",
    )
    detected_objects: List[str] = Field(
        default_factory=list,
        description="List of detected objects in the image",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the analysis",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score of the analysis",
    )


class ImageRecognition:
    """Image recognition using MiniCPM-V model."""

    # Class constants
    DEFAULT_OCR_PROMPT = "Please extract all text content from this image. Include all visible text, maintaining the original structure and formatting as much as possible."
    DEFAULT_DESCRIPTION_PROMPT = "Please provide a detailed description of this image, including objects, people, text, colors, and any other relevant visual information."
    DEFAULT_QA_PROMPT_PREFIX = "Based on the image, please answer/perform the following. Question/Task: "
    MAX_IMAGE_SIZE = (1920, 1080)
    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(self, settings: Optional[ImageRecognitionSettings] = None):
        """Initialize the image recognition module.

        Args:
            settings: Configuration settings for image recognition
        """
        self._settings = settings or ImageRecognitionSettings()
        self._model: Optional[AutoModel] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device = torch.device(self._settings.device)
        self._is_initialized = False

    @property
    def settings(self) -> ImageRecognitionSettings:
        """Get the current settings."""
        return self._settings

    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self._is_initialized

    def _initialize_model(self) -> None:
        """Initialize the MiniCPM-V model and tokenizer."""
        if self._is_initialized:
            return

        try:
            logger.info(f"Loading MiniCPM-V model: {self._settings.model_name}")

            # Get Hugging Face token from settings or environment
            hf_token = self._settings.huggingface_token
            if not hf_token:
                # Try environment variables
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    logger.info("Using Hugging Face token from environment variable")
            else:
                logger.info("Using Hugging Face token from settings")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._settings.model_name,
                trust_remote_code=self._settings.trust_remote_code,
                token=hf_token,
            )

            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": self._settings.trust_remote_code,
                "low_cpu_mem_usage": True,
                "token": hf_token,
                "attn_implementation": "eager",  # Fix for WHISPER_ATTENTION_CLASSES import error
            }

            self._model = AutoModel.from_pretrained(
                self._settings.model_name,
                **model_kwargs,
            ).to(self._device).eval()

            self._is_initialized = True
            logger.info("MiniCPM-V model loaded successfully")

        except Exception as e:
            logger.exception("Failed to initialize MiniCPM-V model")
            raise RuntimeError(f"Failed to initialize MiniCPM-V model: {e}")

    def _preprocess_image(self, image: Union[Image.Image, Path, str, bytes]) -> Image.Image:
        """Preprocess image for model input.

        Args:
            image: Input image in various formats

        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (Path, str)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported image format: {path.suffix}")
            pil_image = Image.open(path)
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert RGBA to RGB if needed
        if pil_image.mode == "RGBA":
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[3])
            pil_image = background
        elif pil_image.mode not in ["RGB", "L"]:
            pil_image = pil_image.convert("RGB")

        # Resize if too large
        if pil_image.size[0] > self.MAX_IMAGE_SIZE[0] or pil_image.size[1] > self.MAX_IMAGE_SIZE[1]:
            pil_image.thumbnail(self.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

        return pil_image

    def _generate_response(self, image: Image.Image, prompt: str) -> str:
        """Generate response from the model.

        Args:
            image: Preprocessed PIL image
            prompt: Text prompt for the model

        Returns:
            Generated text response
        """
        if not self._is_initialized:
            self._initialize_model()

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not properly initialized")

        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Generate response
            response = self._model.chat( # type: ignore
                image=image,
                msgs=messages,
                tokenizer=self._tokenizer,
                context=None,
                sampling=True,
                temperature=self._settings.temperature,
                top_p=self._settings.top_p,
                num_beams=self._settings.num_beams,
                max_new_tokens=self._settings.max_new_tokens,
            )

            return response

        except Exception as e:
            logger.exception("Failed to generate response from MiniCPM-V")
            raise RuntimeError(f"Failed to generate response: {e}")

    def extract_text(self, image: Union[Image.Image, Path, str, bytes]) -> str:
        """Extract text content from an image using OCR.

        Args:
            image: Input image

        Returns:
            Extracted text content
        """
        pil_image = self._preprocess_image(image)
        response = self._generate_response(pil_image, self.DEFAULT_OCR_PROMPT)
        return response

    def describe_image(self, image: Union[Image.Image, Path, str, bytes]) -> str:
        """Generate a natural language description of an image.

        Args:
            image: Input image

        Returns:
            Image description
        """
        pil_image = self._preprocess_image(image)
        response = self._generate_response(pil_image, self.DEFAULT_DESCRIPTION_PROMPT)
        return response

    def answer_question(
        self,
        image: Union[Image.Image, Path, str, bytes],
        question: str,
    ) -> str:
        """Answer a question about an image.

        Args:
            image: Input image
            question: Question to answer about the image

        Returns:
            Answer to the question
        """
        pil_image = self._preprocess_image(image)
        prompt = f"{self.DEFAULT_QA_PROMPT_PREFIX}{question}"
        response = self._generate_response(pil_image, prompt)
        return response

    def caption_for_image(
        self,
        image: Union[Image.Image, Path, str, bytes],
        context: str
    ) -> str:
        """Generate a concise caption for an image.

        Args:
            image: Input image

        Returns:
            Image caption
        """
        return self.answer_question(
            image=image,
            question=f"Provide max 200 character caption for the image taking into account the following context in the page: {context}"
        )
