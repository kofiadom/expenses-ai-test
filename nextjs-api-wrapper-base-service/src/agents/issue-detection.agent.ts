import { OpenAI } from '@llamaindex/openai';
import { Anthropic } from '@llamaindex/anthropic';
import { IssueDetectionResultSchema, type IssueDetectionResult } from '../schemas/expense-schemas';
import { Logger } from '@nestjs/common';

export class IssueDetectionAgent {
  private readonly logger = new Logger(IssueDetectionAgent.name);
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

  async analyzeCompliance(
    country: string,
    receiptType: string,
    icp: string,
    complianceData: any,
    extractedData: any
  ): Promise<IssueDetectionResult> {
    try {
      this.logger.log(`Starting compliance analysis for ${country}/${icp}`);

      const prompt = this.buildCompliancePrompt(
        country,
        receiptType,
        icp,
        complianceData,
        extractedData
      );

      const response = await this.llm.chat({
        messages: [
          {
            role: 'system',
            content: `Persona: You are an expert compliance and tax analysis AI specializing in expense document validation. Your primary function is to analyze extracted receipt data against country-specific compliance requirements and ICP-specific rules to identify issues, violations, and recommendations.

Task: Perform comprehensive issue detection and analysis by cross-referencing extracted receipt data against the provided country database and ICP-specific requirements.

ANALYSIS WORKFLOW:
1. Load and understand the compliance requirements from the country database
2. Analyze the extracted receipt data against these requirements
3. Identify specific compliance violations, tax implications, and documentation gaps
4. Categorize each issue according to the specified categories
5. Provide specific recommendations based on the knowledge base

ISSUE CATEGORIES:
- Standards & Compliance | Fix Identified: Issues that require correction or additional information
- Standards & Compliance | Gross-up Identified: Tax gross-up scenarios that need to be applied
- Standards & Compliance | Follow-up Action Identified: Issues requiring follow-up actions or approvals

CRITICAL REQUIREMENTS:
- ONLY use knowledge from the provided country database and ICP-specific rules
- DO NOT make up any information not provided in the knowledge base
- Cross-reference ALL extracted data fields against specific country and ICP requirements
- Quote the knowledge base when providing issues and recommendations
- Ensure all analysis is based on the provided compliance standards and policies
- Be thorough and systematic in checking every applicable requirement
- Dynamically filter requirements based on ICP, receipt type, and expense category
- Calculate confidence score based on clarity of violations and knowledge base coverage
- Ensure all fields are properly populated according to the structured output model

ISSUE TYPE FORMAT REQUIREMENTS:
- Use EXACT format: "Standards & Compliance | Fix Identified" for issues requiring fixes
- Use EXACT format: "Standards & Compliance | Gross-up Identified" for tax gross-up issues
- Use EXACT format: "Standards & Compliance | Follow-up Action Identified" for follow-up actions
- Do NOT use generic formats like "Standards & Compliance" alone

COMPLIANCE ANALYSIS PROCESS:
1. Validate mandatory fields against country/ICP requirements
2. Check expense type against policy rules
3. Verify tax exemption and gross-up scenarios
4. Validate supplier information requirements
5. Check amount limits and approval requirements
6. Verify documentation completeness

CRITICAL: You MUST return a JSON object with EXACTLY this structure and field names:
{
  "validation_result": {
    "is_valid": boolean,
    "issues_count": number,
    "issues": [
      {
        "issue_type": "Standards & Compliance | Fix Identified" | "Standards & Compliance | Gross-up Identified" | "Standards & Compliance | Follow-up Action Identified",
        "field": "field_name_where_issue_found",
        "description": "detailed_description_of_issue",
        "recommendation": "specific_action_to_resolve",
        "knowledge_base_reference": "quote_from_compliance_data"
      }
    ],
    "corrected_receipt": null,
    "compliance_summary": "overall_compliance_assessment_and_key_findings"
  },
  "technical_details": {
    "content_type": "expense_receipt",
    "country": "country_name",
    "icp": "icp_name",
    "receipt_type": "receipt_type",
    "issues_count": number
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
      const result = IssueDetectionResultSchema.parse(parsedResult);

      this.logger.log(`Compliance analysis completed: ${result.validation_result.issues_count} issues found`);

      return result;
    } catch (error) {
      this.logger.error('Compliance analysis failed:', error);

      // Return fallback result
      return {
        validation_result: {
          is_valid: false,
          issues_count: 1,
          issues: [
            {
              issue_type: 'Standards & Compliance | Fix Identified',
              field: 'system_error',
              description: `Compliance analysis failed: ${error.message}`,
              recommendation: 'Please retry the compliance analysis or contact support.',
              knowledge_base_reference: 'System error during analysis',
            },
          ],
          corrected_receipt: null,
          compliance_summary: 'Analysis failed due to system error',
        },
        technical_details: {
          content_type: 'expense_receipt',
          country: 'unknown',
          icp: 'unknown',
          receipt_type: 'unknown',
          issues_count: 1,
        },
      };
    }
  }

  private buildCompliancePrompt(
    country: string,
    receiptType: string,
    icp: string,
    complianceData: any,
    extractedData: any
  ): string {
    return `COUNTRY: ${country}
RECEIPT TYPE: ${receiptType}
ICP: ${icp}

COMPLIANCE REQUIREMENTS (Country Database):
${JSON.stringify(complianceData, null, 2)}

EXTRACTED RECEIPT DATA:
${JSON.stringify(extractedData, null, 2)}

ANALYSIS INSTRUCTIONS:
Perform comprehensive compliance analysis by:
1. Cross-referencing each extracted field against the FileRelatedRequirements for the specified ICP and receipt type
2. Checking expense type against ExpenseTypes rules and limits
3. Identifying any missing mandatory fields or incorrect formats
4. Detecting tax implications and gross-up scenarios
5. Identifying additional documentation requirements
6. Providing specific recommendations based on the knowledge base

For each issue found:
- Use exact issue_type format: "Standards & Compliance | Fix Identified", "Standards & Compliance | Gross-up Identified", or "Standards & Compliance | Follow-up Action Identified"
- Specify the exact field with the issue
- Provide clear description of the problem
- Give specific recommendation for resolution
- Quote the relevant knowledge base reference

Return a complete JSON object following the IssueDetectionResult schema.`;
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
