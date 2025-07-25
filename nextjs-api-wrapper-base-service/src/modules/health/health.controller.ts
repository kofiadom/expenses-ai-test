import { Controller, Get } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { HealthService } from './health.service';
import { HealthResponseDto } from './dto/health-response.dto';

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(private readonly healthService: HealthService) {}

  @Get()
  @ApiOperation({
    summary: 'Get application health status',
    description: `
      Returns comprehensive health information including:
      - Environment variables (with masked sensitive data)
      - Redis connection status
      - BullMQ queue status
      - System information (memory usage, uptime)
      - Application version and configuration
      
      ⚠️ WARNING: This endpoint exposes configuration data and should be secured in production.
    `
  })
  @ApiResponse({
    status: 200,
    description: 'Health status retrieved successfully',
    type: HealthResponseDto,
  })
  @ApiResponse({
    status: 500,
    description: 'Internal server error while retrieving health status',
  })
  async getHealth(): Promise<HealthResponseDto> {
    return await this.healthService.getHealthStatus();
  }
}
