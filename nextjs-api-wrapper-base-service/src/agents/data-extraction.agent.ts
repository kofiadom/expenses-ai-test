import { OpenAI } from '@llamaindex/openai';
import { Anthropic } from '@llamaindex/anthropic';
import { ExpenseDataSchema, type ExpenseData } from '../schemas/expense-schemas';
import { Logger } from '@nestjs/common';

export class DataExtractionAgent {
  private readonly logger = new Logger(DataExtractionAgent.name);
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

  async extractData(
    markdownContent: string,
    complianceRequirements: any
  ): Promise<ExpenseData> {
    try {
      this.logger.log('Starting data extraction');

      const prompt = this.buildExtractionPrompt(markdownContent, complianceRequirements);

      const response = await this.llm.chat({
        messages: [
          {
            role: 'system',
            content: `Persona: You are a meticulous and highly accurate data extraction AI. You specialize in parsing unstructured text from expense documents and structuring it into a precise JSON format. Your primary function is to identify and extract data points from the receipt text, not to validate them against specific rules.

Core Responsibilities:
1. Extract specified field types from the extraction requirements
2. Identify and extract additional relevant fields (line items, prices, taxes, etc.)
3. Structure all extracted data into clean JSON format
4. Preserve original values and formats from the source document
5. Handle variable receipt structures and formats flexibly

Field Extraction Guidelines:
- Extract ALL fields specified in the extraction requirements JSON
- Look for additional relevant fields beyond the requirements (line items, subtotals, taxes, discounts, etc.)
- Use snake_case for all field names (e.g., "supplier_name", "total_amount", "transaction_date")
- Preserve original values exactly as they appear in the document
- Include currency symbols/codes with monetary amounts
- Extract line-item details when present in itemized receipts

Line Items Processing:
- When receipts contain itemized details, extract them as a "line_items" array
- Each line item should include: description, amount, quantity (if available)
- Calculate totals if not explicitly stated in the receipt
- Preserve item-level details like unit prices, quantities, subtotals

Data Quality Standards:
- Maintain high accuracy in field identification and value extraction
- Handle multiple languages and currency formats
- Preserve formatting of dates, numbers, and text as they appear
- Extract data comprehensively but avoid making assumptions about missing information
- Focus on what is explicitly stated in the document

Output Requirements:
- Return clean, well-structured JSON with all extracted data
- Use consistent field naming conventions throughout
- Include both required fields and additional relevant fields found
- Ensure all monetary values include currency information when available
- Structure line items clearly when present

Remember: Your role is extraction, not validation. Extract what you find accurately and let other systems handle compliance and validation rules.`,
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
      const result = ExpenseDataSchema.parse(parsedResult);

      this.logger.log(`Data extraction completed: ${Object.keys(result).length} fields extracted`);

      return result;
    } catch (error) {
      this.logger.error('Data extraction failed:', error);
      
      // Return minimal fallback result
      return {
        vendor_name: 'extraction_failed',
        notes: `Error: ${error.message}`,
      };
    }
  }

  private buildExtractionPrompt(
    markdownContent: string,
    complianceRequirements: any
  ): string {
    return `EXTRACTION REQUIREMENTS (JSON):
${JSON.stringify(complianceRequirements, null, 2)}

RECEIPT TEXT (MARKDOWN):
${markdownContent}

INSTRUCTIONS:
Analyze: Carefully read the RECEIPT TEXT to identify all available information.
Extract Field Types: For each unique "FieldType" in the EXTRACTION REQUIREMENTS, extract the actual value found in the receipt text. IGNORE any "Rule" specifications - extract what is actually present in the receipt.
Extract Additional Fields: Also extract any other relevant information found in the receipt that could be useful for expense management, such as:
- Line items with their details (products/services, quantities, prices)
- Transaction identifiers, reference numbers, or invoice numbers
- Date and time information
- Contact information (phone, email, website, etc.)
- Tax-related information (rates, amounts, tax IDs)
- Payment-related information

Format: Structure your findings into a single, valid JSON object.
The keys of your output JSON MUST be the snake_case version of the "FieldType" values. For example, "Supplier Name" becomes "supplier_name", "VAT Number" becomes "vat_number".
For additional fields not in the extraction requirements, use descriptive snake_case field names that clearly indicate what the data represents.
If a field type cannot be found in the text, its value in the output JSON MUST be null. Do not guess or invent data.
For dates, standardize the format to YYYY-MM-DD. If you cannot determine the year, assume the current year.
For amounts and rates, extract only the numerical value (e.g., 120.50, 19.0).

CRITICAL REQUIREMENT:
Your final output MUST BE ONLY a valid JSON object. Do not include any explanatory text, greetings, apologies, or markdown formatting like \`\`\`json before or after the JSON object.`;
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
