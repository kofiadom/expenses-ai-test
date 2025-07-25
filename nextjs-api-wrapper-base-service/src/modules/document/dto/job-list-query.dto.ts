import { ApiProperty } from '@nestjs/swagger';
import { IsOptional, IsString, IsNumberString, IsEnum } from 'class-validator';
import { Transform } from 'class-transformer';

export enum JobStatus {
  WAITING = 'waiting',
  ACTIVE = 'active',
  COMPLETED = 'completed',
  FAILED = 'failed',
  DELAYED = 'delayed',
}

export class JobListQueryDto {
  @ApiProperty({
    description: 'Filter by job status',
    enum: JobStatus,
    required: false,
    example: 'completed',
  })
  @IsOptional()
  @IsEnum(JobStatus)
  status?: JobStatus;

  @ApiProperty({
    description: 'Filter by user ID',
    required: false,
    example: 'user123',
  })
  @IsOptional()
  @IsString()
  userId?: string;

  @ApiProperty({
    description: 'Limit number of results (default: 50, max: 100)',
    required: false,
    example: 20,
    minimum: 1,
    maximum: 100,
  })
  @IsOptional()
  @IsNumberString()
  @Transform(({ value }) => parseInt(value))
  limit?: number;

  @ApiProperty({
    description: 'Offset for pagination (default: 0)',
    required: false,
    example: 0,
    minimum: 0,
  })
  @IsOptional()
  @IsNumberString()
  @Transform(({ value }) => parseInt(value))
  offset?: number;
}

export class JobSummaryDto {
  @ApiProperty({
    description: 'Job ID',
    example: 'job_123456789',
  })
  jobId: string;

  @ApiProperty({
    description: 'Job status',
    enum: JobStatus,
    example: 'completed',
  })
  status: JobStatus;

  @ApiProperty({
    description: 'User ID',
    example: 'user123',
  })
  userId: string;

  @ApiProperty({
    description: 'Original filename',
    example: 'medical-report.pdf',
  })
  fileName: string;

  @ApiProperty({
    description: 'Processing language',
    example: 'en',
  })
  language: string;

  @ApiProperty({
    description: 'Job creation timestamp',
    example: '2025-01-15T10:30:00Z',
  })
  createdAt: Date;

  @ApiProperty({
    description: 'Job completion timestamp',
    required: false,
    example: '2025-01-15T10:35:00Z',
  })
  completedAt?: Date;

  @ApiProperty({
    description: 'Processing time in milliseconds',
    required: false,
    example: 45000,
  })
  processingTime?: number;

  @ApiProperty({
    description: 'Error message if failed',
    required: false,
    example: 'Document processing failed: Invalid file format',
  })
  error?: string;
}

export class JobListDataDto {
  @ApiProperty({
    description: 'Array of job summaries',
    type: [JobSummaryDto],
  })
  jobs: JobSummaryDto[];

  @ApiProperty({
    description: 'Total number of jobs matching the filter',
    example: 150,
  })
  total: number;

  @ApiProperty({
    description: 'Current page offset',
    example: 0,
  })
  offset: number;

  @ApiProperty({
    description: 'Number of jobs returned',
    example: 20,
  })
  limit: number;

  @ApiProperty({
    description: 'Whether there are more jobs available',
    example: true,
  })
  hasMore: boolean;
}

export class JobListResponseDto {
  @ApiProperty({
    description: 'Success status',
    example: true,
  })
  success: boolean;

  @ApiProperty({
    description: 'Job list data',
    type: JobListDataDto,
  })
  data: JobListDataDto;
}
