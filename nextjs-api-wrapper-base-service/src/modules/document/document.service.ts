import { Injectable, Logger } from "@nestjs/common";
import { InjectQueue } from "@nestjs/bull";
import { Queue, Job } from "bull";
import { ConfigService } from "@nestjs/config";
import * as fs from "fs";
import * as path from "path";
import {
  DocumentProcessingData,
  ProcessingStatus,
  ProcessingMetrics,
  QUEUE_NAMES,
  JOB_TYPES,
} from "../../types";

@Injectable()
export class DocumentService {
  private readonly logger = new Logger(DocumentService.name);

  constructor(
    @InjectQueue(QUEUE_NAMES.EXPENSE_PROCESSING)
    private expenseQueue: Queue,
    private configService: ConfigService,
  ) {}

  async queueDocumentProcessing(request: {
    file: Express.Multer.File;
    userId: string;
    country: string;
    icp: string;
  }): Promise<{ jobId: string; status: string }> {
    try {
      const { file, userId, country, icp } = request;
      const jobId = userId;
      const uploadPath = this.configService.get("UPLOAD_PATH", "./uploads");

      // Ensure upload directory exists
      if (!fs.existsSync(uploadPath)) {
        fs.mkdirSync(uploadPath, { recursive: true });
      }

      // Save file to permanent location
      const fileName = `${jobId}_${file.originalname}`;
      const filePath = path.join(uploadPath, fileName);

      // Ensure upload directory exists
      if (!fs.existsSync(uploadPath)) {
        fs.mkdirSync(uploadPath, { recursive: true });
      }

      // Handle file path - check if file.path exists (from multer temp storage)
      let sourceFilePath = file.path;

      if (!sourceFilePath) {
        // If no temp path, file might be in buffer - save it directly
        if (file.buffer) {
          this.logger.log(`Saving file buffer to ${filePath}`);
          fs.writeFileSync(filePath, file.buffer);
        } else {
          throw new Error('No file path or buffer available from uploaded file');
        }
      } else {
        // Move file from temp location to permanent location
        this.logger.log(`Moving file from ${sourceFilePath} to ${filePath}`);

        try {
          // Ensure temp file exists
          if (!fs.existsSync(sourceFilePath)) {
            throw new Error(`Temp file not found: ${sourceFilePath}`);
          }

          fs.renameSync(sourceFilePath, filePath);
          this.logger.log(`File successfully moved to ${filePath}`);
        } catch (error) {
          this.logger.error(`Failed to move file: ${error.message}`);
          throw new Error(`Failed to save uploaded file: ${error.message}`);
        }
      }

      const jobData: DocumentProcessingData = {
        jobId,
        filePath,
        fileName: file.originalname,
        userId,
        country,
        icp,
        uploadedAt: new Date(),
      };

      // Add job to expense processing queue
      const job = await this.expenseQueue.add(
        JOB_TYPES.PROCESS_DOCUMENT,
        jobData,
        {
          jobId,
          delay: 0,
          attempts: this.configService.get("MAX_RETRY_ATTEMPTS", 3),
          backoff: {
            type: "exponential",
            delay: 2000,
          },
        }
      );

      this.logger.log(
        `Document processing job queued: ${jobId} for file: ${fileName}`
      );

      return {
        jobId,
        status: "queued",
      };
    } catch (error) {
      this.logger.error("Failed to queue document processing:", error);
      throw error;
    }
  }

  async getProcessingStatus(jobId: string): Promise<ProcessingStatus | null> {
    try {
      // Get all jobs from the expense queue
      const allJobs = await this.expenseQueue.getJobs([
        "waiting",
        "active",
        "completed",
        "failed",
      ]);

      // Find the main document processing job
      const documentJob = allJobs.find(
        (job) =>
          job.data.jobId === jobId && job.name === JOB_TYPES.PROCESS_DOCUMENT
      );

      if (!documentJob) {
        return null;
      }

      // Since all processing is now done in one job, we just need to check the main job
      const jobProgress = documentJob.progress();
      const isCompleted = documentJob.finishedOn !== null;
      const isActive = documentJob.processedOn !== null && !isCompleted;

      // Calculate progress based on job progress
      const progressValue = typeof jobProgress === "number" ? jobProgress : 0;
      const progress = {
        fileClassification: progressValue >= 25,
        dataExtraction: progressValue >= 50,
        issueDetection: progressValue >= 75,
        citationGeneration: progressValue >= 90,
      };

      // Collect results from the single job
      const results: any = {};
      if (documentJob.finishedOn && documentJob.returnvalue) {
        const jobResult = documentJob.returnvalue;
        if (jobResult.data) {
          // Return the complete expense processing result
          Object.assign(results, jobResult.data);
        }
      }

      const status: ProcessingStatus = {
        jobId,
        status: this.getOverallStatus(documentJob, []),
        progress,
        results: Object.keys(results).length > 0 ? results : undefined,
        error: documentJob.failedReason || undefined,
        createdAt: new Date(documentJob.timestamp),
        updatedAt: new Date(documentJob.processedOn || documentJob.timestamp),
      };

      return status;
    } catch (error) {
      this.logger.error(
        `Failed to get processing status for job ${jobId}:`,
        error
      );
      throw error;
    }
  }

  private getOverallStatus(
    mainJob: Job,
    relatedJobs: Job[]
  ): ProcessingStatus["status"] {
    if (mainJob.failedReason || relatedJobs.some((job) => job.failedReason)) {
      return "failed";
    }

    if (mainJob.finishedOn && relatedJobs.every((job) => job.finishedOn)) {
      return "completed";
    }

    if (mainJob.processedOn || relatedJobs.some((job) => job.processedOn)) {
      return "active";
    }

    return "waiting";
  }

  async getProcessingResults(jobId: string): Promise<any | null> {
    try {
      const status = await this.getProcessingStatus(jobId);

      if (!status || status.status !== "completed") {
        return null;
      }

      return status.results;
    } catch (error) {
      this.logger.error(
        `Failed to get processing results for job ${jobId}:`,
        error
      );
      throw error;
    }
  }

  async listJobs(filters: {
    status?: string;
    userId?: string;
    limit: number;
    offset: number;
  }): Promise<{ jobs: ProcessingStatus[]; total: number }> {
    try {
      const { status, userId, limit, offset } = filters;

      // Get jobs from expense processing queue
      const allStates = ["waiting", "active", "completed", "failed", "delayed"];
      const states = status ? [status] : allStates;

      const jobs = await this.expenseQueue.getJobs(
        states as any,
        offset,
        offset + limit - 1
      );

      // Filter for document processing jobs only
      const documentJobs = jobs.filter(
        (job) => job.name === JOB_TYPES.PROCESS_DOCUMENT
      );

      // Filter by userId if provided
      const filteredJobs = userId
        ? documentJobs.filter((job) => job.data.userId === userId)
        : documentJobs;

      // Convert to ProcessingStatus
      const statusPromises = filteredJobs.map((job) =>
        this.getProcessingStatus(job.data.jobId)
      );
      const statuses = (await Promise.all(statusPromises)).filter(
        Boolean
      ) as ProcessingStatus[];

      // Get total count
      const totalCounts = await this.expenseQueue.getJobCounts();
      const total = Object.values(totalCounts).reduce(
        (sum: number, count: number) => sum + count,
        0
      );

      return {
        jobs: statuses,
        total,
      };
    } catch (error) {
      this.logger.error("Failed to list jobs:", error);
      throw error;
    }
  }

  async cancelJob(jobId: string): Promise<boolean> {
    try {
      // Get all jobs related to this jobId
      const allJobs = await this.expenseQueue.getJobs(["waiting", "active"]);
      const relatedJobs = allJobs.filter((job) => job.data.jobId === jobId);

      if (relatedJobs.length === 0) {
        return false;
      }

      // Cancel all related jobs
      for (const job of relatedJobs) {
        await job.remove();
      }

      this.logger.log(
        `Job ${jobId} and ${relatedJobs.length} related jobs cancelled successfully`
      );
      return true;
    } catch (error) {
      this.logger.error(`Failed to cancel job ${jobId}:`, error);
      throw error;
    }
  }

  async getProcessingMetrics(): Promise<ProcessingMetrics> {
    try {
      const counts = await this.expenseQueue.getJobCounts();

      const queueHealth: ProcessingMetrics["queueHealth"] = {
        [QUEUE_NAMES.EXPENSE_PROCESSING]: {
          waiting: counts.waiting || 0,
          active: counts.active || 0,
          completed: counts.completed || 0,
          failed: counts.failed || 0,
        },
      };

      const totalJobs = Object.values(counts).reduce(
        (sum: number, count: number) => sum + count,
        0
      );
      const completedJobs = counts.completed || 0;
      const failedJobs = counts.failed || 0;

      // Calculate average processing time (simplified)
      const recentCompletedJobs = await this.expenseQueue.getJobs(
        ["completed"],
        0,
        99
      );
      const processingTimes = recentCompletedJobs
        .filter((job) => job.finishedOn && job.processedOn)
        .map((job) => job.finishedOn! - job.processedOn!);

      const averageProcessingTime =
        processingTimes.length > 0
          ? processingTimes.reduce((sum, time) => sum + time, 0) /
            processingTimes.length
          : 0;

      return {
        totalJobs,
        completedJobs,
        failedJobs,
        averageProcessingTime,
        queueHealth,
      };
    } catch (error) {
      this.logger.error("Failed to get processing metrics:", error);
      throw error;
    }
  }

  async getHealthStatus(): Promise<any> {
    try {
      const metrics = await this.getProcessingMetrics();

      return {
        service: "Document Processing Service",
        status: "healthy",
        timestamp: new Date().toISOString(),
        metrics,
        queues: {
          [QUEUE_NAMES.EXPENSE_PROCESSING]:
            await this.expenseQueue.getJobCounts(),
        },
      };
    } catch (error) {
      this.logger.error("Health check failed:", error);
      return {
        service: "Document Processing Service",
        status: "unhealthy",
        timestamp: new Date().toISOString(),
        error: error.message,
      };
    }
  }
}
