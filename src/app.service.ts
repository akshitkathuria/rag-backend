import { Injectable } from '@nestjs/common';
import { existsSync, mkdirSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';
import * as fs from 'fs';
import * as path from 'path';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from 'langchain/document';
import { ChatAnthropic } from '@langchain/anthropic';

const SHARED_FOLDER = process.env.SHARED_FOLDER || join(__dirname, '../../shared');
const VECTOR_INDEX_PATH = join(__dirname, '../../vector.index');

@Injectable()
export class AppService {
  constructor() {
    if (!existsSync(SHARED_FOLDER)) {
      mkdirSync(SHARED_FOLDER, { recursive: true });
    }
  }

  saveFile(file: any) {
    const fs = require('fs');
    const dest = join(SHARED_FOLDER, file.originalname);
    fs.writeFileSync(dest, file.buffer);
    return dest;
  }

  listDocuments() {
    const files = readdirSync(SHARED_FOLDER);
    return files.map(name => {
      const filePath = join(SHARED_FOLDER, name);
      const stats = statSync(filePath);
      return {
        name,
        type: extname(name).replace('.', ''),
        size: stats.size,
        uploadDate: stats.birthtime.toISOString(),
      };
    });
  }

  /**
   * Parse and chunk a file into 500-token chunks.
   * Supports PDF, TXT, CSV, JSON.
   * Returns array of {text, source}.
   */
  async parseAndChunkFile(filePath: string) {
    const ext = path.extname(filePath).toLowerCase();
    let text = '';
    if (ext === '.pdf') {
      const pdfParse = require('pdf-parse');
      const dataBuffer = fs.readFileSync(filePath);
      const data = await pdfParse(dataBuffer);
      text = data.text;
    } else if (ext === '.txt') {
      text = fs.readFileSync(filePath, 'utf8');
    } else if (ext === '.csv') {
      const csvParse = require('csv-parse/sync');
      const content = fs.readFileSync(filePath, 'utf8');
      const records = csvParse.parse(content, { columns: false, skip_empty_lines: true });
      text = records.map((row: string[]) => row.join(', ')).join('\n');
    } else if (ext === '.json') {
      const content = fs.readFileSync(filePath, 'utf8');
      const obj = JSON.parse(content);
      text = JSON.stringify(obj, null, 2);
    } else {
      throw new Error('Unsupported file type: ' + ext);
    }
    // Chunking: 500 tokens ~ 2000 characters (approx)
    const chunkSize = 2000;
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
      chunks.push({
        text: text.slice(i, i + chunkSize),
        source: path.basename(filePath),
      });
    }
    return chunks;
  }

  /**
   * Embed and store chunks in FAISS.
   * @param chunks Array of {text, source}
   */
  async embedAndStoreChunks(chunks: {text: string, source: string}[]) {
    const embeddings = new OpenAIEmbeddings();
    let vectorStore: FaissStore;
    // Try to load existing index
    try {
      vectorStore = await FaissStore.load(VECTOR_INDEX_PATH, embeddings);
    } catch {
      vectorStore = await FaissStore.fromDocuments([], embeddings);
    }
    // Convert chunks to Langchain Documents
    const docs = chunks.map(chunk => new Document({
      pageContent: chunk.text,
      metadata: { source: chunk.source }
    }));
    await vectorStore.addDocuments(docs);
    await vectorStore.save(VECTOR_INDEX_PATH);
    return true;
  }

  /**
   * Retrieve top relevant chunks for a query.
   */
  async retrieveRelevantChunks(query: string) {
    const embeddings = new OpenAIEmbeddings();
    let vectorStore: FaissStore;
    try {
      vectorStore = await FaissStore.load(VECTOR_INDEX_PATH, embeddings);
    } catch {
      return [];
    }
    const results = await vectorStore.similaritySearch(query, 4);
    return results.map((doc: Document) => ({
      text: doc.pageContent,
      source: doc.metadata?.source || ''
    }));
  }

  /**
   * Generate answer with Claude via Langchain, given prompt and context.
   */
  async generateAnswerWithContext(prompt: string, contexts: {text: string, source: string}[]) {
    // Build augmented prompt with context
    const contextText = contexts.map((ctx, i) => `Source [${i+1}] (${ctx.source}):\n${ctx.text}`).join('\n\n');
    const fullPrompt = `You are an AI assistant. Use the following context to answer the user's question. Cite the sources in your answer.\n\nContext:\n${contextText}\n\nQuestion: ${prompt}`;
    const model = new ChatAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      model: 'claude-3-haiku-20240307', // or another Claude model you have access to
      maxTokens: 512,
    });
    const response = await model.invoke(fullPrompt);
    return {
      answer: response.content,
      contexts
    };
  }
}
