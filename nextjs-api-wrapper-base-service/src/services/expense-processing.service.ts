import { Injectable, Logger } from '@nestjs/common';
import { FileClassificationAgent } from '../agents/file-classification.agent';
import { DataExtractionAgent } from '../agents/data-extraction.agent';
import { IssueDetectionAgent } from '../agents/issue-detection.agent';
import { CitationGeneratorAgent } from '../agents/citation-generator.agent';
import { ImageQualityAssessmentAgent } from '../agents/image-quality-assessment.agent';
import {
  type FileClassificationResult,
  type ExpenseData,
  type IssueDetectionResult,
  type CitationResult,
  type CompleteProcessingResult,
  type ProcessingTiming,
} from '../schemas/expense-schemas';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class ExpenseProcessingService {
  private readonly logger = new Logger(ExpenseProcessingService.name);
  
  private fileClassificationAgent: FileClassificationAgent;
  private dataExtractionAgent: DataExtractionAgent;
  private issueDetectionAgent: IssueDetectionAgent;
  private citationGeneratorAgent: CitationGeneratorAgent;
  private imageQualityAssessmentAgent: ImageQualityAssessmentAgent;

  constructor() {
    // Force all agents to use Anthropic as default (as requested by user)
    const provider: 'openai' | 'anthropic' = 'anthropic';

    this.logger.log(`Using provider: ${provider} (forced to anthropic)`);

    this.fileClassificationAgent = new FileClassificationAgent(provider);
    this.dataExtractionAgent = new DataExtractionAgent(provider);
    this.issueDetectionAgent = new IssueDetectionAgent(provider);
    this.citationGeneratorAgent = new CitationGeneratorAgent(provider);
    this.imageQualityAssessmentAgent = new ImageQualityAssessmentAgent(provider);
  }

  async processExpenseDocument(
    markdownContent: string,
    filename: string,
    imagePath: string,
    country: string,
    icp: string,
    complianceData: any,
    expenseSchema: any,
    progressCallback?: (stage: string, progress: number) => void
  ): Promise<CompleteProcessingResult> {
    const startTime = Date.now();
    const timing: any = {
      phase_timings: {},
      agent_performance: {},
    };

    try {
      this.logger.log(`Starting complete expense processing for ${filename}`);

      // Phase 0: Image Quality Assessment
      progressCallback?.('imageQualityAssessment', 5);
      this.logger.log('Phase 0: Image Quality Assessment');

      const qualityStart = Date.now();
      const qualityAssessment = await this.imageQualityAssessmentAgent.assessImageQuality(imagePath);
      const qualityEnd = Date.now();
      const formattedQualityAssessment = this.imageQualityAssessmentAgent.formatAssessmentForWorkflow(qualityAssessment, imagePath);

      timing.phase_timings.image_quality_assessment_minutes = ((qualityEnd - qualityStart) / 60000).toFixed(2);
      timing.agent_performance.image_quality_assessment = {
        start_time: new Date(qualityStart).toISOString(),
        end_time: new Date(qualityEnd).toISOString(),
        duration_minutes: ((qualityEnd - qualityStart) / 60000).toFixed(2),
        model_used: 'claude-3-5-sonnet-20241022',
      };

      // Phase 1: File Classification
      progressCallback?.('fileClassification', 15);
      this.logger.log('Phase 1: File Classification');

      const classificationStart = Date.now();
      const classification = await this.fileClassificationAgent.classifyFile(
        markdownContent,
        country,
        expenseSchema
      );
      const classificationEnd = Date.now();

      timing.phase_timings.file_classification_minutes = ((classificationEnd - classificationStart) / 60000).toFixed(2);
      timing.agent_performance.file_classification = {
        start_time: new Date(classificationStart).toISOString(),
        end_time: new Date(classificationEnd).toISOString(),
        duration_minutes: ((classificationEnd - classificationStart) / 60000).toFixed(2),
        model_used: 'claude-3-5-sonnet-20241022',
      };

      progressCallback?.('fileClassification', 25);
      
      // Phase 2: Data Extraction (parallel with classification results)
      progressCallback?.('dataExtraction', 30);
      this.logger.log('Phase 2: Data Extraction');

      const extractionStart = Date.now();
      const extraction = await this.dataExtractionAgent.extractData(
        markdownContent,
        complianceData
      );
      const extractionEnd = Date.now();

      timing.phase_timings.data_extraction_minutes = ((extractionEnd - extractionStart) / 60000).toFixed(2);
      timing.agent_performance.data_extraction = {
        start_time: new Date(extractionStart).toISOString(),
        end_time: new Date(extractionEnd).toISOString(),
        duration_minutes: ((extractionEnd - extractionStart) / 60000).toFixed(2),
        model_used: 'claude-3-5-sonnet-20241022',
      };

      progressCallback?.('dataExtraction', 50);
      
      // Phase 3: Issue Detection
      progressCallback?.('issueDetection', 55);
      this.logger.log('Phase 3: Issue Detection');

      const issueDetectionStart = Date.now();
      const compliance = await this.issueDetectionAgent.analyzeCompliance(
        country,
        classification.expense_type || 'All',
        icp,
        complianceData,
        extraction
      );
      const issueDetectionEnd = Date.now();

      timing.phase_timings.issue_detection_minutes = ((issueDetectionEnd - issueDetectionStart) / 60000).toFixed(2);
      timing.agent_performance.issue_detection = {
        start_time: new Date(issueDetectionStart).toISOString(),
        end_time: new Date(issueDetectionEnd).toISOString(),
        duration_minutes: ((issueDetectionEnd - issueDetectionStart) / 60000).toFixed(2),
        model_used: 'claude-3-5-sonnet-20241022',
      };

      progressCallback?.('issueDetection', 75);

      // Phase 4: Citation Generation
      progressCallback?.('citationGeneration', 80);
      this.logger.log('Phase 4: Citation Generation');

      const citationStart = Date.now();
      const citations = await this.citationGeneratorAgent.generateCitations(
        extraction,
        JSON.stringify(complianceData),
        markdownContent,
        filename
      );
      const citationEnd = Date.now();

      timing.phase_timings.citation_generation_minutes = ((citationEnd - citationStart) / 60000).toFixed(2);
      timing.agent_performance.citation_generation = {
        start_time: new Date(citationStart).toISOString(),
        end_time: new Date(citationEnd).toISOString(),
        duration_minutes: ((citationEnd - citationStart) / 60000).toFixed(2),
        model_used: 'claude-3-5-sonnet-20241022',
      };

      progressCallback?.('citationGeneration', 95);
      
      // Compile final result
      const processingTime = Date.now() - startTime;
      timing.total_processing_time_minutes = (processingTime / 60000).toFixed(2);

      const result: CompleteProcessingResult = {
        image_quality_assessment: formattedQualityAssessment,
        classification,
        extraction,
        compliance,
        citations,
        timing,
        metadata: {
          filename,
          processing_time: processingTime,
          country,
          icp,
          processed_at: new Date().toISOString(),
        },
      };
      
      progressCallback?.('complete', 100);
      this.logger.log(`Complete expense processing finished for ${filename} in ${processingTime}ms`);

      // Save results to file
      await this.saveResultsToFile(filename, result);

      // Save timing results to separate file
      await this.saveTimingToFile(filename, timing);

      return result;
      
    } catch (error) {
      const processingTime = Date.now() - startTime;
      this.logger.error(`Expense processing failed for ${filename}:`, error);
      
      // Return error result
      throw new Error(`Expense processing failed: ${error.message}`);
    }
  }

  async classifyFileOnly(
    markdownContent: string,
    country: string,
    expenseSchema: any
  ): Promise<FileClassificationResult> {
    return this.fileClassificationAgent.classifyFile(markdownContent, country, expenseSchema);
  }

  async extractDataOnly(
    markdownContent: string,
    complianceData: any
  ): Promise<ExpenseData> {
    return this.dataExtractionAgent.extractData(markdownContent, complianceData);
  }

  async analyzeComplianceOnly(
    country: string,
    receiptType: string,
    icp: string,
    complianceData: any,
    extractedData: any
  ): Promise<IssueDetectionResult> {
    return this.issueDetectionAgent.analyzeCompliance(
      country,
      receiptType,
      icp,
      complianceData,
      extractedData
    );
  }

  async generateCitationsOnly(
    extractedData: any,
    extractionRequirements: string,
    markdownContent: string,
    filename: string
  ): Promise<CitationResult> {
    return this.citationGeneratorAgent.generateCitations(
      extractedData,
      extractionRequirements,
      markdownContent,
      filename
    );
  }

  // Health check for all agents
  async healthCheck(): Promise<{ status: string; agents: Record<string, boolean> }> {
    const agents = {
      fileClassification: true,
      dataExtraction: true,
      issueDetection: true,
      citationGeneration: true,
    };

    // Could add actual health checks for each agent here
    // For now, just return healthy status

    const allHealthy = Object.values(agents).every(status => status);
    
    return {
      status: allHealthy ? 'healthy' : 'degraded',
      agents,
    };
  }

  private async saveResultsToFile(filename: string, result: CompleteProcessingResult): Promise<void> {
    try {
      // Create results directory if it doesn't exist
      const resultsDir = path.join(process.cwd(), 'results');
      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
      }

      // Generate output filename based on input filename
      const baseName = path.parse(filename).name;
      const outputFilename = `${baseName}_processed.json`;
      const outputPath = path.join(resultsDir, outputFilename);

      // Save the complete result as JSON
      fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf8');

      this.logger.log(`Results saved to: ${outputPath}`);
    } catch (error) {
      this.logger.error(`Failed to save results for ${filename}:`, error);
      // Don't throw error - saving is optional, don't fail the main process
    }
  }

  private async saveTimingToFile(filename: string, timing: ProcessingTiming): Promise<void> {
    try {
      // Create timing directory if it doesn't exist
      const timingDir = path.join(process.cwd(), 'timing_results');
      if (!fs.existsSync(timingDir)) {
        fs.mkdirSync(timingDir, { recursive: true });
      }

      // Generate timing filename
      const baseFilename = filename.replace(/\.[^/.]+$/, ''); // Remove extension
      const timingFilename = `${baseFilename}_timing.json`;
      const timingFilePath = path.join(timingDir, timingFilename);

      // Add summary information
      const timingWithSummary = {
        ...timing,
        summary: {
          total_time_minutes: timing.total_processing_time_minutes,
          fastest_phase: this.getFastestPhase(timing.phase_timings),
          slowest_phase: this.getSlowestPhase(timing.phase_timings),
          average_phase_time_minutes: this.getAveragePhaseTime(timing.phase_timings),
        },
        generated_at: new Date().toISOString(),
      };

      // Write timing results to file
      fs.writeFileSync(timingFilePath, JSON.stringify(timingWithSummary, null, 2));
      this.logger.log(`Timing results saved to: ${timingFilePath}`);
    } catch (error) {
      this.logger.error('Failed to save timing results to file:', error);
    }
  }

  private getFastestPhase(phaseTimings: any): { phase: string; time_minutes: string } {
    const phases = Object.entries(phaseTimings) as [string, string][];
    const fastest = phases.reduce((min, [phase, time]) =>
      parseFloat(time) < parseFloat(min.time_minutes) ? { phase, time_minutes: time } : min,
      { phase: phases[0][0], time_minutes: phases[0][1] }
    );
    return fastest;
  }

  private getSlowestPhase(phaseTimings: any): { phase: string; time_minutes: string } {
    const phases = Object.entries(phaseTimings) as [string, string][];
    const slowest = phases.reduce((max, [phase, time]) =>
      parseFloat(time) > parseFloat(max.time_minutes) ? { phase, time_minutes: time } : max,
      { phase: phases[0][0], time_minutes: phases[0][1] }
    );
    return slowest;
  }

  private getAveragePhaseTime(phaseTimings: any): string {
    const times = Object.values(phaseTimings) as string[];
    const average = times.reduce((sum, time) => sum + parseFloat(time), 0) / times.length;
    return average.toFixed(2);
  }
}
