import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';
import { HealthResponseDto } from './dto/health-response.dto';

@Injectable()
export class HealthService {
  private startTime = Date.now();

  constructor(private configService: ConfigService) {}

  async getHealthStatus(): Promise<HealthResponseDto> {
    
    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: this.getUptime(),
      environment: this.getEnvironmentVariables(),
      connections: await this.getConnectionStatus(),
      system: this.getSystemInfo(),
      warning: 'WARNING: This endpoint exposes sensitive configuration data. Use with caution.'
    };
  }

  private getEnvironmentVariables() {
    return {
      NODE_ENV: this.configService.get('NODE_ENV', 'not set'),
      PORT: this.configService.get('PORT', 'not set'),
      OPENAI_API_KEY: this.maskApiKey(this.configService.get('OPENAI_API_KEY')),
      ANTHROPIC_KEY: this.maskApiKey(this.configService.get('ANTHROPIC_KEY')),
      LLAMAINDEX_API_KEY: this.maskApiKey(this.configService.get('LLAMAINDEX_API_KEY')),
      MAX_FILE_SIZE: this.configService.get('MAX_FILE_SIZE', 'not set'),
      UPLOAD_PATH: this.configService.get('UPLOAD_PATH', 'not set'),
      QUEUE_CONCURRENCY: this.configService.get('QUEUE_CONCURRENCY', 'not set'),
      MAX_RETRY_ATTEMPTS: this.configService.get('MAX_RETRY_ATTEMPTS', 'not set'),
      JOB_TIMEOUT: this.configService.get('JOB_TIMEOUT', 'not set'),
      ENABLE_SWAGGER: this.configService.get('ENABLE_SWAGGER', 'not set'),
      ENABLE_THROTTLING: this.configService.get('ENABLE_THROTTLING', 'not set'),
      THROTTLE_TTL: this.configService.get('THROTTLE_TTL', 'not set'),
      THROTTLE_LIMIT: this.configService.get('THROTTLE_LIMIT', 'not set'),
    };
  }

  private async getConnectionStatus() {
    const redisStatus = await this.checkRedisConnection();
    const bullmqStatus = this.checkBullMQStatus();

    return {
      redis: redisStatus,
      bullmq: bullmqStatus
    };
  }

  private async checkRedisConnection() {
    const startTime = Date.now();
    let redis: Redis;

    try {
      redis = new Redis({
        host: this.configService.get('REDIS_HOST', 'localhost'),
        port: this.configService.get('REDIS_PORT', 6379),
        password: this.configService.get('REDIS_PASSWORD') || undefined,
        db: this.configService.get('REDIS_DB', 0),
        connectTimeout: 5000,
        lazyConnect: true,
      });

      await redis.connect();
      await redis.ping();
      
      const responseTime = Date.now() - startTime;
      
      return {
        status: 'connected',
        host: this.configService.get('REDIS_HOST', 'localhost'),
        port: parseInt(this.configService.get('REDIS_PORT', '6379')),
        database: parseInt(this.configService.get('REDIS_DB', '0')),
        responseTime: `${responseTime}ms`
      };
    } catch (error) {
      return {
        status: 'disconnected',
        host: this.configService.get('REDIS_HOST', 'localhost'),
        port: parseInt(this.configService.get('REDIS_PORT', '6379')),
        database: parseInt(this.configService.get('REDIS_DB', '0')),
        responseTime: 'N/A',
        error: error.message
      };
    } finally {
      if (redis) {
        redis.disconnect();
      }
    }
  }

  private checkBullMQStatus() {
    try {
      // Since BullMQ is configured in the app, we assume it's working if Redis is working
      // In a real scenario, you might want to inject the Queue and check its status
      return {
        status: 'configured',
        queues: ['document-processing'], // Based on your app structure
      };
    } catch (error) {
      return {
        status: 'error',
        queues: [],
        error: error.message
      };
    }
  }

  private getSystemInfo() {
    const memoryUsage = process.memoryUsage();
    const totalMemory = memoryUsage.heapTotal;
    const usedMemory = memoryUsage.heapUsed;
    const percentage = ((usedMemory / totalMemory) * 100).toFixed(1);

    return {
      memory: {
        used: this.formatBytes(usedMemory),
        total: this.formatBytes(totalMemory),
        percentage: `${percentage}%`
      }
    };
  }

  private getUptime(): string {
    const uptimeMs = Date.now() - this.startTime;
    const seconds = Math.floor(uptimeMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  private maskApiKey(apiKey: string): string {
    if (!apiKey || apiKey === 'not set') {
      return 'not set';
    }
    
    if (apiKey.length <= 12) {
      return '***masked***';
    }
    
    const start = apiKey.substring(0, 8);
    const end = apiKey.substring(apiKey.length - 4);
    return `${start}...${end}`;
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  }
}
