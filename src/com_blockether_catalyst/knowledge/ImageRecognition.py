"""
Image recognition module using OpenAI compatible models for visual understanding.

This module provides image recognition capabilities for knowledge extraction,
including OCR, visual question answering, and image content understanding.
"""

import base64
import io
import logging
from textwrap import dedent
from typing import Optional

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageRecognitionSettings(BaseModel):
    """Settings for image recognition."""

    base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Base URL for the OpenAI-compatible server",
    )
    model_id: str = Field(
        default="qwen2.5-vl-3b-instruct",
        description="Model ID to use on the server",
    )
    api_key: Optional[str] = Field(
        default="none",
        description="API key for authentication (if required)",
    )
    max_tokens: int = Field(
        default=256,
        description="Maximum tokens for the model response",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature for response generation",
    )
    system_prompt: str = Field(
        default=dedent(
            """
            You are an expert captioning assistant.
            Describe the image in one short, clear sentence suitable as a caption.
            Do not include unnecessary details or context. Be specific and concise.
        """
        ),
        description="System prompt to guide the model's behavior",
    )


class ImageRecognition:
    def __init__(self, settings: Optional[ImageRecognitionSettings] = None):
        """Initialize the image recognition module.

        Args:
            settings: Configuration settings for image recognition
        """
        self._settings = settings or ImageRecognitionSettings()
        self._llm: Optional[OpenAI] = OpenAI(api_key=self._settings.api_key, base_url=self._settings.base_url)

    @property
    def settings(self) -> ImageRecognitionSettings:
        """Get the current settings."""
        return self._settings

    def _image_to_base64_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI.

        Args:
            image: PIL Image object

        Returns:
            Base64 data URI string
        """
        # Save image to bytes buffer
        buffer = io.BytesIO()

        # Determine the format based on image mode
        if image.mode == "RGBA":
            format_str = "PNG"
            mime_type = "image/png"
        else:
            format_str = "JPEG"
            mime_type = "image/jpeg"

        # Save the image to buffer
        image.save(buffer, format=format_str)
        buffer.seek(0)

        # Encode to base64
        base64_data = base64.b64encode(buffer.read()).decode("utf-8")

        # Return as data URI
        return f"data:{mime_type};base64,{base64_data}"

    def _generate_response(self, image: Image.Image, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate response from the model.

        Args:
            image: Preprocessed PIL image
            prompt: Text prompt for the model

        Returns:
            Generated text response
        """
        if self._llm is None:
            raise RuntimeError("Model not properly initialized")

        if not system_prompt:
            system_prompt = self._settings.system_prompt

        try:
            # Convert image to base64 data URI
            image_data_uri = self._image_to_base64_data_uri(image)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            # Generate response using the model
            response = self._llm.chat.completions.create(
                model=self._settings.model_id,
                messages=messages,
                max_tokens=self._settings.max_tokens,
                temperature=self._settings.temperature,
            )

            text_response = response.choices[0].message.content

            if not text_response:
                raise ValueError("No content in response")

            return text_response

        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")

    def answer_question(self, image: Image.Image, question: str, system_prompt: Optional[str] = None) -> str:
        """Answer a question about an image.

        Args:
            image: Input image
            question: Question to answer about the image

        Returns:
            Answer to the question
        """
        response = self._generate_response(image, question, system_prompt)
        return response

    def _normalize_caption(self, caption: str) -> str:
        # Remove leading/trailing quotes
        caption = caption.strip(" \"'")

        # Remove leading 'Caption', 'Caption:', or 'caption:'
        for prefix in ["Caption:", "caption:", "Caption"]:
            if caption.startswith(prefix):
                caption = caption[len(prefix) :].lstrip()

        # Strip whitespace
        caption = caption.strip()

        # Ensure it ends with a period
        if caption and not caption.endswith("."):
            caption += "."

        return caption

    def caption_for_image(self, image: Image.Image, context: str, system_prompt: Optional[str] = None) -> str:
        """Generate a concise caption for an image.

        Args:
            image: Input image

        Returns:
            Image caption
        """
        return self._normalize_caption(
            self.answer_question(
                image=image,
                question=f"Generate the caption for the image taking into account the following context: {context}",
                system_prompt=system_prompt,
            )
        )
