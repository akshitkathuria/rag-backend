import { Controller, Get, Post, UploadedFile, UseInterceptors, Body } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { AppService } from './app.service';
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

@Controller('api')
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: any) {
    const filePath = this.appService.saveFile(file);
    const chunks = await this.appService.parseAndChunkFile(filePath);
    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HF_API_KEY,
      model: "sentence-transformers/all-MiniLM-L6-v2"
    });
    await this.appService.embedAndStoreChunks(chunks);
    return { success: true, name: file.originalname };
  }

  @Get('documents')
  getDocuments() {
    return this.appService.listDocuments();
  }

  @Post('chat')
  async chat(@Body() body: { prompt: string }) {
    const contexts = await this.appService.retrieveRelevantChunks(body.prompt);
    const result = await this.appService.generateAnswerWithContext(body.prompt, contexts);
    return result;
  }
}
