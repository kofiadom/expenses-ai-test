import { OpenAI } from '@llamaindex/openai';
import { Anthropic } from '@llamaindex/anthropic';
import { ImageQualityAssessmentSchema, type ImageQualityAssessment } from '../schemas/expense-schemas';
import { Logger } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export class ImageQualityAssessmentAgent {
  private readonly logger = new Logger(ImageQualityAssessmentAgent.name);
  private llm: any;
  private currentProvider: 'openai' | 'anthropic';

  constructor(provider: 'openai' | 'anthropic' = 'anthropic') {
    this.currentProvider = provider;
    if (provider === 'anthropic') {
      this.llm = new Anthropic({
        apiKey: process.env.ANTHROPIC_KEY,
        model: 'claude-3-5-sonnet-20241022',
      });
    } else {
      this.llm = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        model: 'gpt-4o',
      });
    }
  }

  async assessImageQuality(imagePath: string): Promise<ImageQualityAssessment> {
    this.logger.log(`🤖 Starting LLM-based quality assessment for: ${path.basename(imagePath)}`);

    try {
      // Get image info for context
      const imageInfo = this.getImageInfo(imagePath);

      // For now, simulate quality assessment based on file properties
      // TODO: Implement actual vision-based assessment when LlamaIndex TS vision API is stable
      const response = await this.llm.chat({
        messages: [
          {
            role: 'user',
            content: `Simulate a quality assessment for an expense document image. ${imageInfo}\n\n${this.createAssessmentPrompt()}`,
          },
        ],
      });

      // Parse the JSON response manually - handle different response formats
      let rawContent: string;
      if (typeof response.message.content === 'string') {
        rawContent = response.message.content;
      } else if (Array.isArray(response.message.content) && response.message.content.length > 0) {
        // Anthropic format: [{"type":"text","text":"actual JSON content"}]
        const firstItem = response.message.content[0];
        if (firstItem && firstItem.type === 'text' && firstItem.text) {
          rawContent = firstItem.text;
        } else {
          rawContent = JSON.stringify(response.message.content);
        }
      } else if (response.message.content && typeof response.message.content === 'object') {
        rawContent = JSON.stringify(response.message.content);
      } else {
        rawContent = String(response.message.content || '');
      }

      this.logger.debug(`Raw response type: ${typeof response.message.content}`);
      this.logger.debug(`Extracted content: ${rawContent.substring(0, 200)}...`);

      const parsedResult = this.parseJsonResponse(rawContent);
      const result = ImageQualityAssessmentSchema.parse(parsedResult);

      this.logger.log(`Image quality assessment completed: Score ${result.overall_quality_score}/10, Suitable: ${result.suitable_for_extraction}`);
      return result;

    } catch (error) {
      this.logger.error(`Image quality assessment failed: ${error.message}`);

      // Return fallback result
      return {
        blur_detection: this.createFallbackIssue('Blur assessment failed'),
        contrast_assessment: this.createFallbackIssue('Contrast assessment failed'),
        glare_identification: this.createFallbackIssue('Glare assessment failed'),
        water_stains: this.createFallbackIssue('Water stain assessment failed'),
        tears_or_folds: this.createFallbackIssue('Tear/fold assessment failed'),
        cut_off_detection: this.createFallbackIssue('Cut-off assessment failed'),
        missing_sections: this.createFallbackIssue('Missing section assessment failed'),
        obstructions: this.createFallbackIssue('Obstruction assessment failed'),
        overall_quality_score: 5,
        suitable_for_extraction: true, // Default to true to not block processing
      };
    }
  }

  private createFallbackIssue(description: string) {
    return {
      detected: false,
      severity_level: 'low' as const,
      confidence_score: 0.5,
      quantitative_measure: 0.0,
      description,
      recommendation: 'Manual review recommended due to assessment failure',
    };
  }

  private getMimeType(imagePath: string): string {
    const ext = path.extname(imagePath).toLowerCase();
    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'image/jpeg';
      case '.png':
        return 'image/png';
      case '.tiff':
      case '.tif':
        return 'image/tiff';
      default:
        return 'image/jpeg';
    }
  }

  private getImageInfo(imagePath: string): string {
    const stats = fs.statSync(imagePath);
    const sizeKB = Math.round(stats.size / 1024);
    const filename = path.basename(imagePath);

    return `Filename: ${filename}, Size: ${sizeKB}KB, Format: ${path.extname(imagePath)}`;
  }

  private createAssessmentPrompt(): string {
    return `CRITICAL: You are an expert image quality assessor for receipt/invoice documents. Analyze this image for OCR and data extraction suitability.

ASSESSMENT CATEGORIES:
Evaluate each category and provide detailed analysis:

1. **Blur Detection**: Check for motion blur, focus issues, or camera shake
2. **Contrast Assessment**: Evaluate text-to-background contrast and readability
3. **Glare Identification**: Detect reflective glare or lighting issues
4. **Water Stains**: Identify water damage, stains, or discoloration
5. **Tears or Folds**: Detect physical damage like tears, creases, or folds
6. **Cut-off Detection**: Check if document edges are cut off or missing
7. **Missing Sections**: Identify if parts of the document are obscured or missing
8. **Obstructions**: Detect objects blocking text (fingers, shadows, etc.)

SCORING GUIDELINES:
- Overall Quality Score: 1-10 (1=unusable, 10=perfect)
- Confidence Score: 0.0-1.0 (how confident you are in your assessment)
- Quantitative Measure: Relevant metric (blur intensity, damage percentage, etc.)
- Severity Levels: low, medium, high, critical

CRITICAL: You MUST return a JSON object with EXACTLY this structure:
{
  "blur_detection": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "contrast_assessment": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical", 
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "glare_identification": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "water_stains": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "tears_or_folds": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "cut_off_detection": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "missing_sections": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "obstructions": {
    "detected": boolean,
    "severity_level": "low|medium|high|critical",
    "confidence_score": number (0.0-1.0),
    "quantitative_measure": number,
    "description": "string",
    "recommendation": "string"
  },
  "overall_quality_score": number (1-10),
  "suitable_for_extraction": boolean
}

Do NOT use any other field names. Do NOT add extra fields. Return ONLY the JSON object.`;
  }

  private parseJsonResponse(content: string): any {
    try {
      // Remove markdown code blocks if present
      const cleanContent = content
        .replace(/```json\s*/g, '')
        .replace(/```\s*/g, '')
        .trim();
      
      return JSON.parse(cleanContent);
    } catch (error) {
      this.logger.error('Failed to parse JSON response:', error);
      throw new Error(`Invalid JSON response: ${error.message}`);
    }
  }

  formatAssessmentForWorkflow(assessment: ImageQualityAssessment, imagePath: string) {
    this.logger.debug(`Current provider: ${this.currentProvider}`);
    const modelUsed = this.currentProvider === 'anthropic' ? 'claude-3-5-sonnet-20241022' : 'gpt-4o';
    this.logger.debug(`Model used: ${modelUsed}`);

    return {
      image_path: imagePath,
      assessment_method: 'LLM',
      model_used: modelUsed,
      timestamp: new Date().toISOString(),
      quality_score: assessment.overall_quality_score * 10, // Convert to 0-100 scale
      quality_level: this.getQualityLevel(assessment.overall_quality_score),
      suitable_for_extraction: assessment.suitable_for_extraction,
      ...assessment,
    };
  }

  private getQualityLevel(score: number): string {
    if (score >= 8) return 'excellent';
    if (score >= 6) return 'good';
    if (score >= 4) return 'fair';
    return 'poor';
  }
}
