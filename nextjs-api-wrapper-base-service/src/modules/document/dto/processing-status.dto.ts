import { ApiProperty } from '@nestjs/swagger';

export class LabParameterMatchingProgressDto {
  @ApiProperty({
    description: 'Total number of lab parameters to process',
    example: 25,
  })
  total: number;

  @ApiProperty({
    description: 'Number of completed lab parameter matches',
    example: 15,
  })
  completed: number;

  @ApiProperty({
    description: 'Completion percentage',
    example: 60,
  })
  percentage: number;
}

export class ProcessingProgressDto {
  @ApiProperty({
    description: 'Document summary completion status',
    example: true,
  })
  documentSummary: boolean;

  @ApiProperty({
    description: 'Physician matching completion status',
    example: true,
  })
  physicianMatching: boolean;

  @ApiProperty({
    description: 'Facility matching completion status',
    example: false,
  })
  facilityMatching: boolean;

  @ApiProperty({
    description: 'Lab parameter matching progress details',
    type: LabParameterMatchingProgressDto,
  })
  labParameterMatching: LabParameterMatchingProgressDto;
}

export class ProcessingStatusDto {
  @ApiProperty({
    description: 'Job ID',
    example: 'job_123456789',
  })
  jobId: string;

  @ApiProperty({
    description: 'Current job status',
    enum: ['waiting', 'active', 'completed', 'failed', 'delayed'],
    example: 'active',
  })
  status: 'waiting' | 'active' | 'completed' | 'failed' | 'delayed';

  @ApiProperty({
    description: 'Processing progress details',
    type: ProcessingProgressDto,
  })
  progress: ProcessingProgressDto;

  @ApiProperty({
    description: 'Partial results (if available)',
    required: false,
    example: {
      summary: { /* document summary data */ },
      physicianMatch: { /* physician match data */ },
    },
  })
  results?: {
    summary?: any;
    physicianMatch?: any;
    facilityMatch?: any;
    labMatches?: any[];
  };

  @ApiProperty({
    description: 'Error message (if failed)',
    required: false,
    example: 'Document processing failed: Invalid file format',
  })
  error?: string;

  @ApiProperty({
    description: 'Job creation timestamp',
    example: '2025-01-15T10:30:00Z',
  })
  createdAt: Date;

  @ApiProperty({
    description: 'Last update timestamp',
    example: '2025-01-15T10:35:00Z',
  })
  updatedAt: Date;
}

export class ProcessingStatusResponseDto {
  @ApiProperty({
    description: 'Success status',
    example: true,
  })
  success: boolean;

  @ApiProperty({
    description: 'Processing status data',
    type: ProcessingStatusDto,
  })
  data: ProcessingStatusDto;
}
