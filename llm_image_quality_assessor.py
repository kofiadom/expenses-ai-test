import os
import json
import base64
import time
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from PIL import Image
from agno.utils.log import logger


class QualityIssue(BaseModel):
    """Individual quality issue assessment"""
    detected: bool = Field(description="Whether the issue was detected (True/False)")
    severity_level: str = Field(description="Severity level: 'low', 'medium', 'high', 'critical'")
    confidence_score: float = Field(description="Confidence in detection (0.0-1.0)", ge=0.0, le=1.0)
    quantitative_measure: float = Field(description="Quantitative measure when applicable (e.g., blur intensity, damage percentage)")
    description: str = Field(description="Short description of the finding in one sentence")
    recommendation: str = Field(description="Simple recommendation to address the issue")


class LLMImageQualityAssessment(BaseModel):
    """Complete LLM-based image quality assessment results"""
    blur_detection: QualityIssue = Field(description="Blur detection assessment")
    contrast_assessment: QualityIssue = Field(description="Contrast quality assessment")
    glare_identification: QualityIssue = Field(description="Glare detection assessment")
    water_stains: QualityIssue = Field(description="Water stain damage detection")
    tears_or_folds: QualityIssue = Field(description="Physical tears or fold detection")
    cut_off_detection: QualityIssue = Field(description="Edge cut-off detection")
    missing_sections: QualityIssue = Field(description="Missing content sections detection")
    obstructions: QualityIssue = Field(description="Obstruction detection")

    overall_quality_score: int = Field(description="Overall quality score from 1-10", ge=1, le=10)
    suitable_for_extraction: bool = Field(description="Whether image is suitable for OCR/data extraction")


class LLMImageQualityAssessor:
    """LLM-based image quality assessor for integration with expense processing workflow"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize LLM image quality assessor
        
        Args:
            api_key: Anthropic API key (if None, will use environment variable)
            model: Claude model to use for assessment
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        logger.info(f"🤖 Initialized LLM Image Quality Assessor with model: {model}")
    
    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode image to base64 string and return with correct media type"""
        try:
            # Get the correct media type
            media_type = self.get_media_type_from_image(image_path)
            
            # Handle format conversion if needed
            with Image.open(image_path) as img:
                if img.format and img.format.lower() not in ['jpeg', 'jpg', 'png', 'webp', 'gif']:
                    # Convert to RGB and save as JPEG
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    import io
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr = img_byte_arr.getvalue()
                    return base64.b64encode(img_byte_arr).decode('utf-8'), "image/jpeg"
                else:
                    # Use original file
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8'), media_type
                        
        except Exception as e:
            raise Exception(f"Error encoding image {image_path}: {str(e)}")
    
    def get_media_type_from_image(self, image_path: str) -> str:
        """Determine the correct media type by reading the actual image format"""
        try:
            with Image.open(image_path) as img:
                format_lower = img.format.lower() if img.format else None
                if format_lower in ['jpeg', 'jpg']:
                    return "image/jpeg"
                elif format_lower == 'png':
                    return "image/png"
                elif format_lower == 'webp':
                    return "image/webp"
                elif format_lower in ['gif']:
                    return "image/gif"
                else:
                    return "image/jpeg"
        except Exception:
            return "image/jpeg"
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get basic image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "size_mb": round(os.path.getsize(image_path) / (1024 * 1024), 2)
                }
        except Exception as e:
            return {"error": str(e)}
    
    def create_assessment_prompt(self) -> str:
        """Create the professional prompt for image quality assessment"""
        return """You are an expert image quality analyst specializing in receipt and invoice document assessment. Your task is to thoroughly analyze the provided receipt/invoice image and assess its quality across multiple dimensions before OCR/data extraction processing.

ANALYSIS REQUIREMENTS:

1. **BLUR DETECTION**: Examine text sharpness, edge definition, and overall focus quality. Look for motion blur, camera shake, or out-of-focus areas that would impair text recognition.
   - Provide quantitative_measure: blur intensity (0.0=sharp, 1.0=extremely blurry)
   - Assess severity_level and confidence_score

2. **CONTRAST ASSESSMENT**: Evaluate the contrast between text and background. Check for adequate differentiation that enables clear text recognition.
   - Provide quantitative_measure: contrast ratio assessment (0.0=poor, 1.0=excellent)
   - Consider lighting conditions and background uniformity

3. **GLARE IDENTIFICATION**: Detect bright spots, reflections, or glare that obscure text or important document areas. Look for overexposed regions.
   - Provide quantitative_measure: percentage of image affected by glare (0.0-1.0)
   - Identify specific areas where glare impacts readability

4. **WATER STAIN DETECTION**: Identify water damage including discoloration, staining, warping effects, or color distortions that affect document readability.
   - Provide quantitative_measure: percentage of document affected (0.0-1.0)
   - Assess impact on text legibility

5. **TEARS OR FOLDS DETECTION**: Look for physical damage like tears, creases, folds, or wrinkles that may cause text distortion or information loss.
   - Provide quantitative_measure: severity of physical damage (0.0=none, 1.0=severe)
   - Count visible fold lines or tear areas

6. **CUT-OFF DETECTION**: Check if document edges are cut off or if the image frame excludes important document portions.
   - Provide quantitative_measure: percentage of document potentially cut off (0.0-1.0)
   - Identify which edges are affected

7. **MISSING SECTIONS**: Identify if parts of the receipt/invoice are missing, incomplete, or not captured in the image.
   - Provide quantitative_measure: estimated percentage of content missing (0.0-1.0)
   - Consider typical receipt structure

8. **OBSTRUCTIONS**: Detect any objects, fingers, shadows, or other elements that block or obscure document content.
   - Provide quantitative_measure: percentage of document obscured (0.0-1.0)
   - Identify types of obstructions

ASSESSMENT CRITERIA:
- For each quality issue, determine if it's detected (True/False)
- Assign severity_level: 'low', 'medium', 'high', 'critical'
- Provide confidence_score (0.0-1.0) for your detection confidence
- Include quantitative_measure for measurable aspects
- Provide a concise, factual description in one sentence
- Give practical recommendations
- Assign an overall quality score (1-10, where 10 is perfect quality)
- Determine if the image is suitable for OCR/data extraction

IMPORTANT GUIDELINES:
- Focus specifically on receipt/invoice characteristics (structured text, tables, line items, totals)
- Be thorough but practical in your assessment
- Consider the impact on automated text extraction systems
- Prioritize issues that would significantly impair data extraction accuracy
- Use quantitative measures to provide objective assessments where possible

Analyze the provided image and return your assessment in the following JSON structure:

{
  "blur_detection": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (blur intensity 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "contrast_assessment": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (contrast quality 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "glare_identification": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (percentage affected 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "water_stains": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (percentage affected 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "tears_or_folds": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (damage severity 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "cut_off_detection": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (percentage cut off 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "missing_sections": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (percentage missing 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "obstructions": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": float (0.0-1.0),
    "quantitative_measure": float (percentage obscured 0.0-1.0),
    "description": "string",
    "recommendation": "string"
  },
  "overall_quality_score": integer (1-10),
  "suitable_for_extraction": boolean
}

Return only the JSON response with no additional text."""

    async def assess_image_quality(self, image_path: str) -> LLMImageQualityAssessment:
        """
        Assess image quality using LLM vision capabilities

        Args:
            image_path: Path to the image file

        Returns:
            LLMImageQualityAssessment object with detailed assessment results
        """
        logger.info(f"🤖 Starting LLM-based quality assessment for: {os.path.basename(image_path)}")
        assessment_start_time = time.time()

        try:
            # Encode image and get metadata
            base64_image, media_type = self.encode_image(image_path)
            img_info = self.get_image_info(image_path)

            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please analyze this receipt/invoice image for quality assessment. Image info: {img_info}\n\n{self.create_assessment_prompt()}"
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }

            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")

                    result = await response.json()
                    content = result['content'][0]['text']

                    # Extract JSON from response
                    try:
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if start_idx != -1 and end_idx != 0:
                            json_str = content[start_idx:end_idx]
                            assessment_data = json.loads(json_str)
                        else:
                            assessment_data = json.loads(content)
                    except json.JSONDecodeError:
                        raise Exception(f"Failed to parse JSON response: {content}")

                    assessment = LLMImageQualityAssessment(**assessment_data)

                    assessment_time = time.time() - assessment_start_time
                    logger.info(f"✅ LLM quality assessment complete - Score: {assessment.overall_quality_score}/10, "
                              f"Suitable: {assessment.suitable_for_extraction}, Time: {assessment_time:.2f}s")

                    return assessment

        except Exception as e:
            logger.error(f"❌ LLM quality assessment failed for {image_path}: {str(e)}")
            raise Exception(f"LLM quality assessment failed: {str(e)}")

    def assess_image_quality_sync(self, image_path: str) -> LLMImageQualityAssessment:
        """
        Synchronous version of assess_image_quality using requests instead of aiohttp

        Args:
            image_path: Path to the image file

        Returns:
            LLMImageQualityAssessment object with detailed assessment results
        """
        logger.info(f"🤖 Starting LLM-based quality assessment for: {os.path.basename(image_path)}")
        assessment_start_time = time.time()

        try:
            # Encode image and get metadata
            base64_image, media_type = self.encode_image(image_path)
            img_info = self.get_image_info(image_path)

            # Prepare API request
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            payload = {
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please analyze this receipt/invoice image for quality assessment. Image info: {img_info}\n\n{self.create_assessment_prompt()}"
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }

            # Make synchronous API request
            response = requests.post(self.base_url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            result = response.json()
            content = result['content'][0]['text']

            # Extract JSON from response
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    assessment_data = json.loads(json_str)
                else:
                    assessment_data = json.loads(content)
            except json.JSONDecodeError:
                raise Exception(f"Failed to parse JSON response: {content}")

            assessment = LLMImageQualityAssessment(**assessment_data)

            assessment_time = time.time() - assessment_start_time
            logger.info(f"✅ LLM quality assessment complete - Score: {assessment.overall_quality_score}/10, "
                      f"Suitable: {assessment.suitable_for_extraction}, Time: {assessment_time:.2f}s")

            return assessment

        except Exception as e:
            logger.error(f"❌ LLM quality assessment failed for {image_path}: {str(e)}")
            raise Exception(f"LLM quality assessment failed: {str(e)}")

    def format_assessment_for_workflow(self, assessment: LLMImageQualityAssessment, image_path: str) -> Dict:
        """
        Format LLM assessment results in the enhanced format with quantitative parameters

        Args:
            assessment: LLMImageQualityAssessment object
            image_path: Path to the assessed image

        Returns:
            Dictionary in enhanced format with LLM assessment results
        """
        # Create the enhanced format with all quantitative parameters
        result = {
            'image_path': image_path,
            'assessment_method': 'LLM',
            'model_used': self.model,
            'timestamp': datetime.now().isoformat(),

            # Enhanced format fields with quantitative parameters
            'blur_detection': assessment.blur_detection.model_dump(),
            'contrast_assessment': assessment.contrast_assessment.model_dump(),
            'glare_identification': assessment.glare_identification.model_dump(),
            'water_stains': assessment.water_stains.model_dump(),
            'tears_or_folds': assessment.tears_or_folds.model_dump(),
            'cut_off_detection': assessment.cut_off_detection.model_dump(),
            'missing_sections': assessment.missing_sections.model_dump(),
            'obstructions': assessment.obstructions.model_dump(),
            'overall_quality_score': assessment.overall_quality_score,
            'suitable_for_extraction': assessment.suitable_for_extraction
        }

        return result

    def _determine_quality_level(self, score: int) -> str:
        """Determine quality level based on score"""
        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"
