import { ApiProperty } from '@nestjs/swagger';

export class EnvironmentVariablesDto {
  @ApiProperty({ example: 'development' })
  NODE_ENV: string;

  @ApiProperty({ example: '3000' })
  PORT: string;

  @ApiProperty({ example: 'sk-proj-z...sBtZ', description: 'Masked for security' })
  OPENAI_API_KEY: string;

  @ApiProperty({ example: 'sk-ant-a...GAAA', description: 'Masked for security' })
  ANTHROPIC_KEY: string;

  @ApiProperty({ example: 'llx-YbTb...Ma8', description: 'Masked for security' })
  LLAMAINDEX_API_KEY: string;

  @ApiProperty({ example: '50MB' })
  MAX_FILE_SIZE: string;

  @ApiProperty({ example: './uploads' })
  UPLOAD_PATH: string;

  @ApiProperty({ example: '10' })
  QUEUE_CONCURRENCY: string;

  @ApiProperty({ example: '3' })
  MAX_RETRY_ATTEMPTS: string;

  @ApiProperty({ example: '300000' })
  JOB_TIMEOUT: string;

  @ApiProperty({ example: 'true' })
  ENABLE_SWAGGER: string;

  @ApiProperty({ example: 'true' })
  ENABLE_THROTTLING: string;

  @ApiProperty({ example: '60' })
  THROTTLE_TTL: string;

  @ApiProperty({ example: '100' })
  THROTTLE_LIMIT: string;
}

export class RedisConnectionDto {
  @ApiProperty({ example: 'connected' })
  status: string;

  @ApiProperty({ example: 'localhost' })
  host: string;

  @ApiProperty({ example: 6379 })
  port: number;

  @ApiProperty({ example: 0 })
  database: number;

  @ApiProperty({ example: '2ms' })
  responseTime: string;

  @ApiProperty({ example: null, required: false })
  error?: string;
}

export class BullMQConnectionDto {
  @ApiProperty({ example: 'connected' })
  status: string;

  @ApiProperty({ example: ['document-processing'] })
  queues: string[];

  @ApiProperty({ example: null, required: false })
  error?: string;
}

export class ConnectionsDto {
  @ApiProperty({ type: RedisConnectionDto })
  redis: RedisConnectionDto;

  @ApiProperty({ type: BullMQConnectionDto })
  bullmq: BullMQConnectionDto;
}

export class MemoryDto {
  @ApiProperty({ example: '45.2 MB' })
  used: string;

  @ApiProperty({ example: '512 MB' })
  total: string;

  @ApiProperty({ example: '8.8%' })
  percentage: string;
}

export class SystemDto {
  @ApiProperty({ type: MemoryDto })
  memory: MemoryDto;
}

export class HealthResponseDto {
  @ApiProperty({ example: 'healthy' })
  status: string;

  @ApiProperty({ example: '2025-01-15T12:39:00Z' })
  timestamp: string;

  @ApiProperty({ example: '2h 15m 30s' })
  uptime: string;



  @ApiProperty({ type: EnvironmentVariablesDto })
  environment: EnvironmentVariablesDto;

  @ApiProperty({ type: ConnectionsDto })
  connections: ConnectionsDto;

  @ApiProperty({ type: SystemDto })
  system: SystemDto;

  @ApiProperty({ example: 'WARNING: This endpoint exposes sensitive configuration data. Use with caution.' })
  warning: string;
}
