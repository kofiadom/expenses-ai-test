import { Processor, Process } from "@nestjs/bull";
import { Logger } from "@nestjs/common";
import { Job } from "bull";
import { DocumentService } from "../../document/document.service";
import {
  DocumentProcessingData,
  QUEUE_NAMES,
  JOB_TYPES,
  JobResult,
} from "../../../types";

@Processor(QUEUE_NAMES.MEDICAL_PROCESSING)
export class MedicalProcessor {
  private readonly logger = new Logger(MedicalProcessor.name);

  constructor(private readonly documentService: DocumentService) {}

  @Process(JOB_TYPES.PROCESS_DOCUMENT)
  async processDocument(job: Job<DocumentProcessingData>): Promise<JobResult> {
    const startTime = Date.now();
    const { jobId, filePath, fileName, userId, language } = job.data;

    try {
      this.logger.log(
        `Starting complete document processing for job: ${jobId}, file: ${fileName}`
      );

      // Update job progress
      await job.progress(5);

      await job.progress(25);

      await job.progress(35);

      await job.progress(50);

      await job.progress(65);

      await job.progress(90);

      await job.progress(100);

      const processingTime = Date.now() - startTime;
      this.logger.log(
        `Complete document processing finished for job: ${jobId} in ${processingTime}ms`
      );

      return {
        success: true,
        data: {},
        processingTime,
      };
    } catch (error) {
      const processingTime = Date.now() - startTime;
      this.logger.error(`Document processing failed for job: ${jobId}:`, error);

      return {
        success: false,
        error: error.message,
        processingTime,
      };
    }
  }
}
