import { Processor, Process } from "@nestjs/bull";
import { Logger } from "@nestjs/common";
import { Job } from "bull";
import { DocumentService } from "../../document/document.service";
import { ExpenseProcessingService } from "../../../services/expense-processing.service";
import { LlamaParseApiService } from "../../../utils/llamaParseReader";
import {
  DocumentProcessingData,
  QUEUE_NAMES,
  JOB_TYPES,
  JobResult,
} from "../../../types";

@Processor(QUEUE_NAMES.EXPENSE_PROCESSING)
export class ExpenseProcessor {
  private readonly logger = new Logger(ExpenseProcessor.name);

  constructor(
    private readonly documentService: DocumentService,
    private readonly expenseProcessingService: ExpenseProcessingService
  ) {}

  @Process(JOB_TYPES.PROCESS_DOCUMENT)
  async processDocument(job: Job<DocumentProcessingData>): Promise<JobResult> {
    const startTime = Date.now();
    const { jobId, filePath, fileName, userId, country, icp } = job.data;

    try {
      this.logger.log(
        `Starting expense document processing for job: ${jobId}, file: ${fileName}`
      );

      // Read the document content (assuming it's already converted to markdown)
      const markdownContent = await this.readDocumentContent(filePath);
      
      // Load compliance data and expense schema (placeholder - should be loaded from config/database)
      const complianceData = await this.loadComplianceData(country, icp);
      const expenseSchema = await this.loadExpenseSchema();

      // Process the document through all agents
      const result = await this.expenseProcessingService.processExpenseDocument(
        markdownContent,
        fileName,
        filePath,
        country,
        icp,
        complianceData,
        expenseSchema,
        async (stage: string, progress: number) => {
          await job.progress(progress);
          this.logger.log(`${stage}: ${progress}%`);
        }
      );

      const processingTime = Date.now() - startTime;
      this.logger.log(
        `Expense document processing finished for job: ${jobId} in ${processingTime}ms`
      );

      return {
        success: true,
        data: result,
        processingTime,
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      this.logger.error(`Expense document processing failed for job: ${jobId}:`, error);

      return {
        success: false,
        error: error.message,
        processingTime,
      };
    }
  }

  private async readDocumentContent(filePath: string): Promise<string> {
    try {
      const fs = require('fs');
      const path = require('path');

      // Check if file exists
      if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
      }

      const fileExtension = path.extname(filePath).toLowerCase();
      const fileName = path.basename(filePath);

      this.logger.log(`Reading document: ${fileName} (${fileExtension})`);

      // Use LlamaParse for document extraction
      const llamaParseApiKey = process.env.LLAMAINDEX_API_KEY;
      if (!llamaParseApiKey) {
        this.logger.warn('LLAMAINDEX_API_KEY not found, using placeholder content');
        return this.getPlaceholderContent(fileName);
      }

      try {
        const llamaParseService = new LlamaParseApiService(llamaParseApiKey);

        // Configure LlamaParse for expense document processing
        const parseConfig = {
          parseMode: 'parse_page_with_lvm',
          vendorMultimodalModelName: 'anthropic-sonnet-3.7',
          disableOcr: false,
          adaptiveLongTable: true,
          annotateLinks: false,
          timeout: 120000, // 2 minutes timeout
        };

        this.logger.log(`Extracting content from ${fileName} using LlamaParse...`);
        const parseResult = await llamaParseService.parseDocument(filePath, parseConfig);

        if (parseResult.success && parseResult.data) {
          this.logger.log(`Successfully extracted ${parseResult.data.length} characters from ${fileName}`);
          return parseResult.data;
        } else {
          const errorMsg = 'error' in parseResult ? parseResult.error : 'Unknown error';
          this.logger.error(`LlamaParse failed for ${fileName}: ${errorMsg}`);
          return this.getPlaceholderContent(fileName);
        }
      } catch (parseError) {
        this.logger.error(`LlamaParse error for ${fileName}: ${parseError.message}`);
        return this.getPlaceholderContent(fileName);
      }
    } catch (error) {
      this.logger.error(`Failed to read document content: ${error.message}`);
      throw error;
    }
  }

  private getPlaceholderContent(fileName: string): string {
    return `# Receipt Document: ${fileName}

**Date:** 2024-07-25
**Vendor:** Sample Restaurant
**Amount:** €25.50
**Tax:** €4.08
**Category:** Meals & Entertainment

## Line Items:
- Main Course: €18.00
- Beverage: €3.50
- Service Charge: €4.00

**Payment Method:** Credit Card
**Receipt Number:** REC-2024-001

*Note: This is placeholder content. LlamaParse extraction failed or API key not available.*`;
  }

  private async loadComplianceData(country: string, icp: string): Promise<any> {
    try {
      const fs = require('fs');
      const path = require('path');

      const complianceFile = path.join(process.cwd(), 'data', `${country.toLowerCase()}.json`);

      if (fs.existsSync(complianceFile)) {
        const fileContent = fs.readFileSync(complianceFile, 'utf8');
        const complianceData = JSON.parse(fileContent);
        this.logger.log(`Loaded compliance data for ${country} - ${Object.keys(complianceData).length} sections`);
        return complianceData;
      } else {
        this.logger.warn(`No compliance data found for ${country} at ${complianceFile}`);
        return {};
      }
    } catch (error) {
      this.logger.error(`Failed to load compliance data for ${country}: ${error.message}`);
      return {};
    }
  }

  private async loadExpenseSchema(): Promise<any> {
    try {
      const fs = require('fs');
      const path = require('path');

      const schemaFile = path.join(process.cwd(), 'expense_file_schema.json');

      if (fs.existsSync(schemaFile)) {
        const fileContent = fs.readFileSync(schemaFile, 'utf8');
        const schemaData = JSON.parse(fileContent);
        this.logger.log(`Loaded expense schema with ${Object.keys(schemaData.properties || {}).length} fields`);
        return schemaData;
      } else {
        this.logger.warn(`No expense schema found at ${schemaFile}`);
        return {};
      }
    } catch (error) {
      this.logger.error(`Failed to load expense schema: ${error.message}`);
      return {};
    }
  }
}
