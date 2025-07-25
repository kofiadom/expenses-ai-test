import { OpenAI } from '@llamaindex/openai';
import { Anthropic } from '@llamaindex/anthropic';
import { FileClassificationResultSchema, type FileClassificationResult } from '../schemas/expense-schemas';
import { Logger } from '@nestjs/common';

export class FileClassificationAgent {
  private readonly logger = new Logger(FileClassificationAgent.name);
  private llm: any;



  constructor(provider: 'openai' | 'anthropic' = 'anthropic') {
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

  async classifyFile(
    markdownContent: string,
    expectedCountry: string,
    expenseSchema: any
  ): Promise<FileClassificationResult> {
    try {
      this.logger.log('Starting file classification');

      const prompt = this.buildClassificationPrompt(markdownContent, expectedCountry, expenseSchema);

      const response = await this.llm.chat({
        messages: [
          {
            role: 'system',
            content: `Persona: You are an expert file classification AI specializing in expense document analysis. Your primary function is to determine if a file contains expense-related content and classify it appropriately.

Task: Analyze the provided text to determine:
1. Whether this is an expense document (Y/N)
2. If it's an expense, classify the expense type
3. Identify the document language and confidence level
4. Verify location consistency

CLASSIFICATION CRITERIA:

STEP 1: EXPENSE IDENTIFICATION (SCHEMA-BASED)
First determine: Is this file an expense? (Y/N)

Use the provided EXPENSE FILE SCHEMA to identify expense documents based on field presence.

Look for each schema field in the document. If you find 3-4 or more fields, it's an expense document.

REQUIRED FOR EXPENSE CLASSIFICATION:
- Evidence of payment completed (not just booking/reservation)
- Actual amounts charged/paid
- Payment confirmation or receipt of transaction

EXPENSE TYPE CLUSTERS:
- flights: Flight tickets, airline receipts, boarding passes
- meals: Restaurant receipts, food purchases, catering services
- accommodation: Hotel bills, lodging expenses, vacation rentals
- telecommunications: Phone bills, internet services, communication expenses
- travel: Ground transportation, trains, buses, taxis, car rentals
- training: Educational courses, conferences, workshops, seminars
- mileage: Vehicle expenses, fuel, parking, tolls
- entertainment: Client entertainment, business meals with clients

LANGUAGE IDENTIFICATION:
- Identify the primary language of the document
- Provide confidence score (0-100) for language identification
- Consider mixed-language documents and identify the dominant language

LOCATION VALIDATION:
- Extract location information from the document (country, city, region)
- Compare with the expected location provided
- Set location_match to true if they align, false if they don't match

ERROR CATEGORIES:
- unreadable_content: Document is corrupted, illegible, or cannot be processed
- insufficient_content: Document lacks enough information for classification
- classification_error: System error during classification process
- language_detection_error: Unable to determine document language
- location_extraction_error: Unable to extract location information

PROCESSING WORKFLOW:
1. First check if content is readable and processable
2. Identify language and calculate confidence score
3. Determine if content represents an expense document
4. If expense, classify the expense type cluster
5. Extract document location information
6. Compare document location with expected location
7. Set appropriate error flags if any issues found

CRITICAL REQUIREMENTS:
- Be conservative in classification - when in doubt, mark as not an expense
- Follow the exact error categories specified
- Provide clear reasoning for your decision
- Ensure all fields are properly populated according to the structured output model

CRITICAL: You MUST return a JSON object with EXACTLY this structure and field names:
{
  "is_expense": boolean,
  "expense_type": string | null,
  "language": string,
  "language_confidence": number (0-100),
  "document_location": string,
  "expected_location": string,
  "location_match": boolean,
  "error_type": string | null,
  "error_message": string | null,
  "classification_confidence": number (0-100),
  "reasoning": string,
  "schema_field_analysis": {
    "fields_found": string[],
    "fields_missing": string[],
    "total_fields_found": number,
    "expense_identification_reasoning": string
  }
}

Do NOT use any other field names. Do NOT add extra fields. Return ONLY the JSON object.`,
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      // Parse the JSON response manually since structured output isn't working as expected
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
      const result = FileClassificationResultSchema.parse(parsedResult);

      this.logger.log(`File classification completed: ${result.is_expense ? 'EXPENSE' : 'NOT_EXPENSE'} - ${result.expense_type} (${result.language})`);

      return result;
    } catch (error) {
      this.logger.error('File classification failed:', error);
      
      // Return fallback result
      return {
        is_expense: false,
        expense_type: null,
        language: 'unknown',
        language_confidence: 0,
        document_location: 'unknown',
        expected_location: 'unknown',
        location_match: false,
        error_type: 'classification_error',
        error_message: error.message,
        classification_confidence: 0,
        reasoning: `Classification failed due to error: ${error.message}`,
        schema_field_analysis: {
          fields_found: [],
          fields_missing: [],
          total_fields_found: 0,
          expense_identification_reasoning: 'Classification failed due to system error',
        },
      };
    }
  }

  private buildClassificationPrompt(
    markdownContent: string,
    expectedCountry: string,
    expenseSchema: any
  ): string {
    // Create schema field descriptions for the prompt
    let schemaFieldsDescription = "";
    for (const [fieldName, fieldInfo] of Object.entries(expenseSchema?.properties || {})) {
      const title = (fieldInfo as any)?.title || fieldName;
      const description = (fieldInfo as any)?.description || "";
      schemaFieldsDescription += `\n**${fieldName}** (${title}):\n${description}\n`;
    }

    return `EXPENSE FILE SCHEMA FIELDS:
${schemaFieldsDescription}

DOCUMENT TEXT TO ANALYZE:
${markdownContent}

EXPECTED LOCATION: ${expectedCountry || "Not specified"}

ANALYSIS INSTRUCTIONS:
1. Carefully examine the document text for each of the 8 schema fields listed above
2. For each field, determine if it is PRESENT or ABSENT in the document
3. Use the field descriptions and recognition patterns to guide your analysis
4. Count the total number of fields found
5. Apply the expense identification logic (3-4+ fields = expense)
6. Provide detailed reasoning citing the exact fields found/missing
7. Identify the document language and provide confidence score
8. Extract location information and compare with expected location
9. Classify expense type if determined to be an expense document
10. Set appropriate error flags if any processing issues occur

EXPENSE IDENTIFICATION LOGIC:
- Files with 3-4+ matching schema fields should be considered expenses
- Be conservative in classification - when in doubt, mark as not an expense
- Focus on core expense indicators: supplier info, amounts, dates, tax details

SCHEMA FIELD ANALYSIS REQUIREMENTS:
- List all fields found in the document with specific examples
- List all fields missing from the document
- Provide total count of fields found
- Give detailed reasoning for expense identification decision citing exact fields

CRITICAL: You MUST return a JSON object with EXACTLY this structure and field names:
{
  "is_expense": boolean,
  "expense_type": string | null,
  "language": string,
  "language_confidence": number (0-100),
  "document_location": string,
  "expected_location": string,
  "location_match": boolean,
  "error_type": string | null,
  "error_message": string | null,
  "classification_confidence": number (0-100),
  "reasoning": string,
  "schema_field_analysis": {
    "fields_found": string[],
    "fields_missing": string[],
    "total_fields_found": number,
    "expense_identification_reasoning": string
  }
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
}
