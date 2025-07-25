import {
  Controller,
  Post,
  Get,
  Delete,
  Param,
  Query,
  UploadedFile,
  UseInterceptors,
  Body,
  HttpStatus,
  HttpException,
  UsePipes,
  ValidationPipe,
} from "@nestjs/common";
import { FileInterceptor } from "@nestjs/platform-express";
import {
  ApiTags,
  ApiOperation,
  ApiResponse,
  ApiConsumes,
  ApiParam,
  ApiQuery,
  ApiBody,
} from "@nestjs/swagger";
import { DocumentService } from "./document.service";
import {
  ProcessDocumentResponseDto,
  ProcessingStatusResponseDto,
  ErrorResponseDto,
  ValidationErrorResponseDto,
} from "./dto";

@ApiTags("documents")
@Controller("documents")
export class DocumentController {
  constructor(private readonly documentService: DocumentService) {}

  @Post("process")
  @UseInterceptors(
    FileInterceptor("file", {
      limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
      },
      fileFilter: (req, file, cb) => {
        const allowedMimes = [
          "application/pdf",
          "image/png",
          "image/jpeg",
          "image/jpg",
        ];
        if (allowedMimes.includes(file.mimetype)) {
          cb(null, true);
        } else {
          cb(
            new Error(
              "Invalid file type. Only PDF, PNG, JPG, and JPEG files are allowed."
            ),
            false
          );
        }
      },
    })
  )
  @UsePipes(new ValidationPipe({ transform: true }))
  @ApiOperation({
    summary: "Upload and process a medical document",
    description:
      "Upload a medical document (PDF, PNG, JPG, JPEG) for AI-powered processing. The document will be analyzed to extract patient information, physician details, medical facility information, and lab test results with parameter matching.",
  })
  @ApiConsumes("multipart/form-data")
  @ApiBody({
    description: "Medical document upload with processing parameters",
    schema: {
      type: "object",
      properties: {
        file: {
          type: "string",
          format: "binary",
          description: "Medical document file (PDF, PNG, JPG, JPEG, max 10MB)",
        },
        userId: {
          type: "string",
          description: "User ID for processing context",
          example: "user123",
        },
        language: {
          type: "string",
          description: "Processing language (default: en)",
          enum: ["en", "he"],
          default: "en",
        },
      },
      required: ["file", "userId"],
    },
  })
  @ApiResponse({
    status: 201,
    description: "Document processing job created successfully",
    type: ProcessDocumentResponseDto,
    schema: {
      example: {
        success: true,
        message: "Document processing job created successfully",
        data: {
          jobId: "job_123456789",
          status: "waiting",
          createdAt: "2025-01-15T10:30:00Z",
        },
      },
    },
  })
  @ApiResponse({
    status: 400,
    description: "Invalid file or request parameters",
    type: ErrorResponseDto,
    schema: {
      example: {
        success: false,
        message:
          "Invalid file type. Only PDF, PNG, JPG, and JPEG files are allowed.",
        statusCode: 400,
        timestamp: "2025-01-15T10:30:00Z",
        path: "/documents/process",
      },
    },
  })
  @ApiResponse({
    status: 413,
    description: "File too large (max 10MB)",
    type: ErrorResponseDto,
    schema: {
      example: {
        success: false,
        message: "File size exceeds the 10MB limit",
        statusCode: 413,
        timestamp: "2025-01-15T10:30:00Z",
        path: "/documents/process",
      },
    },
  })
  @ApiResponse({
    status: 422,
    description: "Validation error",
    type: ValidationErrorResponseDto,
    schema: {
      example: {
        success: false,
        message: "Validation failed",
        statusCode: 422,
        timestamp: "2025-01-15T10:30:00Z",
        path: "/documents/process",
        errors: ["userId should not be empty", "file must be provided"],
      },
    },
  })
  @ApiResponse({
    status: 500,
    description: "Internal server error",
    type: ErrorResponseDto,
    schema: {
      example: {
        success: false,
        message:
          "Failed to queue document processing: Service temporarily unavailable",
        statusCode: 500,
        timestamp: "2025-01-15T10:30:00Z",
        path: "/documents/process",
      },
    },
  })
  async processDocument(
    @UploadedFile() file: Express.Multer.File,
    @Body() body: { userId: string; language?: string }
  ) {
    if (!file) {
      throw new HttpException("No file uploaded", HttpStatus.BAD_REQUEST);
    }

    if (!body.userId) {
      throw new HttpException("userId is required", HttpStatus.BAD_REQUEST);
    }

    try {
      const result = await this.documentService.queueDocumentProcessing({
        file,
        userId: body.userId,
        language: body.language || "en",
      });

      return {
        success: true,
        message: "Document processing job created successfully",
        data: result,
      };
    } catch (error) {
      throw new HttpException(
        `Failed to queue document processing: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Get("status/:jobId")
  @ApiOperation({
    summary: "Get processing status for a job",
    description:
      "Retrieve the current processing status and progress for a specific job. Returns detailed information about each processing stage including document summary, physician matching, facility matching, and lab parameter matching progress.",
  })
  @ApiParam({
    name: "jobId",
    description: "Unique job identifier returned from the process endpoint",
    example: "job_123456789",
  })
  @ApiResponse({
    status: 200,
    description: "Job status retrieved successfully",
    type: ProcessingStatusResponseDto,
    schema: {
      example: {
        success: true,
        data: {
          jobId: "job_123456789",
          status: "active",
          progress: {
            documentSummary: true,
            physicianMatching: true,
            facilityMatching: false,
            labParameterMatching: {
              total: 25,
              completed: 15,
              percentage: 60,
            },
          },
          results: {
            summary: {
              /* document summary data */
            },
            physicianMatch: {
              /* physician match data */
            },
          },
          createdAt: "2025-01-15T10:30:00Z",
          updatedAt: "2025-01-15T10:33:00Z",
        },
      },
    },
  })
  @ApiResponse({
    status: 404,
    description: "Job not found",
    type: ErrorResponseDto,
    schema: {
      example: {
        success: false,
        message: "Job not found",
        statusCode: 404,
        timestamp: "2025-01-15T10:30:00Z",
        path: "/documents/status/job_123456789",
      },
    },
  })
  @ApiResponse({
    status: 500,
    description: "Internal server error",
    type: ErrorResponseDto,
  })
  async getProcessingStatus(@Param("jobId") jobId: string) {
    try {
      const status = await this.documentService.getProcessingStatus(jobId);

      if (!status) {
        throw new HttpException("Job not found", HttpStatus.NOT_FOUND);
      }

      return {
        success: true,
        data: status,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        `Failed to get job status: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Get("results/:jobId")
  @ApiOperation({ summary: "Get final processing results for a completed job" })
  @ApiParam({ name: "jobId", description: "Job ID to get results for" })
  @ApiResponse({
    status: 200,
    description: "Job results retrieved successfully",
  })
  @ApiResponse({ status: 404, description: "Job not found or not completed" })
  async getProcessingResults(@Param("jobId") jobId: string) {
    try {
      const results = await this.documentService.getProcessingResults(jobId);

      if (!results) {
        throw new HttpException(
          "Job not found or not completed",
          HttpStatus.NOT_FOUND
        );
      }

      return {
        success: true,
        data: results,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        `Failed to get job results: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Get("jobs")
  @ApiOperation({ summary: "List processing jobs with optional filtering" })
  @ApiQuery({
    name: "status",
    required: false,
    description: "Filter by job status",
  })
  @ApiQuery({
    name: "userId",
    required: false,
    description: "Filter by user ID",
  })
  @ApiQuery({
    name: "limit",
    required: false,
    description: "Limit number of results",
  })
  @ApiQuery({
    name: "offset",
    required: false,
    description: "Offset for pagination",
  })
  @ApiResponse({ status: 200, description: "Jobs list retrieved successfully" })
  async listJobs(
    @Query("status") status?: string,
    @Query("userId") userId?: string,
    @Query("limit") limit?: string,
    @Query("offset") offset?: string
  ) {
    try {
      const jobs = await this.documentService.listJobs({
        status,
        userId,
        limit: limit ? parseInt(limit) : 50,
        offset: offset ? parseInt(offset) : 0,
      });

      return {
        success: true,
        data: jobs,
      };
    } catch (error) {
      throw new HttpException(
        `Failed to list jobs: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Delete("jobs/:jobId")
  @ApiOperation({ summary: "Cancel a processing job" })
  @ApiParam({ name: "jobId", description: "Job ID to cancel" })
  @ApiResponse({ status: 200, description: "Job cancelled successfully" })
  @ApiResponse({ status: 404, description: "Job not found" })
  async cancelJob(@Param("jobId") jobId: string) {
    try {
      const result = await this.documentService.cancelJob(jobId);

      if (!result) {
        throw new HttpException(
          "Job not found or cannot be cancelled",
          HttpStatus.NOT_FOUND
        );
      }

      return {
        success: true,
        message: "Job cancelled successfully",
        data: result,
      };
    } catch (error) {
      if (error instanceof HttpException) {
        throw error;
      }
      throw new HttpException(
        `Failed to cancel job: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Get("metrics")
  @ApiOperation({ summary: "Get processing metrics and queue health" })
  @ApiResponse({ status: 200, description: "Metrics retrieved successfully" })
  async getMetrics() {
    try {
      const metrics = await this.documentService.getProcessingMetrics();

      return {
        success: true,
        data: metrics,
      };
    } catch (error) {
      throw new HttpException(
        `Failed to get metrics: ${error.message}`,
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  @Get("health")
  @ApiOperation({ summary: "Health check endpoint" })
  @ApiResponse({ status: 200, description: "Service health status" })
  async healthCheck() {
    try {
      const health = await this.documentService.getHealthStatus();

      return {
        success: true,
        data: health,
      };
    } catch (error) {
      throw new HttpException(
        `Health check failed: ${error.message}`,
        HttpStatus.SERVICE_UNAVAILABLE
      );
    }
  }
}
