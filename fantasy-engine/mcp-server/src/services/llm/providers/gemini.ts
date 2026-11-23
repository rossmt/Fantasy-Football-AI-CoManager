import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import { BaseLLMProvider } from './base.js';
import { LLMMessage, LLMTool, LLMResponse, LLMConfig } from '../types.js';

// Tried in order when the configured model is overloaded, out of quota or
// retired. Override with GEMINI_FALLBACK_MODELS.
const DEFAULT_GEMINI_FALLBACK_MODELS = 'gemini-3.6-flash,gemini-3-flash-preview,gemini-3.5-flash-lite';


export class GeminiProvider extends BaseLLMProvider {
  private client: GoogleGenerativeAI;
  
  constructor(config: LLMConfig) {
    super(config);
    this.client = new GoogleGenerativeAI(config.api_key);
  }
  
  get name(): string {
    return 'Google Gemini';
  }
  
  get models(): string[] {
    return [
      'gemini-3.7-flash',
      'gemini-3.6-flash',
      'gemini-3.5-flash-lite',
      'gemini-3.1-pro-preview'
    ];
  }
  
  async chat(
    messages: LLMMessage[],
    options?: {
      tools?: LLMTool[];
      max_tokens?: number;
      temperature?: number;
      tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
    }
  ): Promise<LLMResponse> {
    const candidates = this.candidateModels();
    let lastError: any;

    for (let i = 0; i < candidates.length; i++) {
      const modelName = candidates[i];
      try {
        // Retry the configured model a couple of times for short blips, then
        // move on quickly: an overloaded model tends to stay overloaded.
        const response = await this.withRetry(
          () => this.chatWithModel(modelName, messages, options),
          i === 0 ? 2 : 1
        );
        if (modelName !== this.config.model) {
          console.warn(`⚠️ Switched Gemini model from ${this.config.model} to ${modelName} for the rest of this run`);
          this.config.model = modelName;
        }
        return response;
      } catch (error: any) {
        lastError = error;
        const next = candidates[i + 1];
        if (!next || !GeminiProvider.isModelUnavailable(error)) throw error;
        console.warn(`⚠️ Gemini model ${modelName} unavailable (${error.status}), falling back to ${next}...`);
      }
    }
    throw lastError;
  }

  /**
   * Configured model first, then GEMINI_FALLBACK_MODELS (comma-separated),
   * without duplicates. Set GEMINI_FALLBACK_MODELS to an empty string to
   * disable fallback.
   */
  private candidateModels(): string[] {
    const raw = process.env.GEMINI_FALLBACK_MODELS ?? DEFAULT_GEMINI_FALLBACK_MODELS;
    const fallbacks = raw.split(',').map(m => m.trim()).filter(Boolean);
    return [...new Set([this.config.model, ...fallbacks])];
  }

  /**
   * Errors that mean "this model can't serve us right now" rather than a
   * problem with the request: overloaded (503), server error (500), out of
   * quota for this model (429) or model retired/unknown (404).
   */
  private static isModelUnavailable(error: any): boolean {
    return [404, 429, 500, 503].includes(error?.status);
  }

  private async chatWithModel(
    modelName: string,
    messages: LLMMessage[],
    options?: {
      tools?: LLMTool[];
      max_tokens?: number;
      temperature?: number;
      tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
    }
  ): Promise<LLMResponse> {
      const startTime = Date.now();
      
      // Get the model
      const model = this.client.getGenerativeModel({
        model: modelName,
        generationConfig: {
          temperature: options?.temperature || this.config.temperature || 0.7,
          maxOutputTokens: options?.max_tokens || this.config.max_tokens || 4000,
          // Gemini 3.x models think by default and count thinking tokens against
          // maxOutputTokens, which can exhaust the budget before any visible
          // text is produced. Keep thinking minimal so short responses aren't
          // silently truncated to empty content.
          thinkingConfig: { thinkingLevel: 'low' },
        } as any,
        safetySettings: [
          {
            category: HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          },
          {
            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          },
        ],
      });
      
      // Convert messages to Gemini format
      const geminiMessages = this.convertMessagesToGemini(messages);
      
      // Handle function calling
      let tools: any;
      if (options?.tools && options.tools.length > 0) {
        tools = [{
          functionDeclarations: options.tools.map(tool => ({
            name: tool.name,
            description: tool.description,
            parameters: {
              ...tool.input_schema,
              type: 'OBJECT' as const
            }
          }))
        }];
      }
      
      // Start chat session
      const chat = model.startChat({
        history: geminiMessages.slice(0, -1), // All messages except the last
        tools: tools
      });
      
      // Send the last message
      const lastMessage = geminiMessages[geminiMessages.length - 1];
      const result = await chat.sendMessage(lastMessage.parts[0].text);
      const responseTime = Date.now() - startTime;
      
      const response = await result.response;
      let content = response.text();
      
      // Handle function calls
      let tool_calls;
      const functionCalls = (response as any).functionCalls;
      if (functionCalls && Array.isArray(functionCalls) && functionCalls.length > 0) {
        tool_calls = functionCalls.map((fc: any) => ({
          name: fc.name,
          arguments: fc.args
        }));
      }
      
      return {
        content,
        tool_calls,
        usage: {
          input_tokens: response.usageMetadata?.promptTokenCount,
          output_tokens: response.usageMetadata?.candidatesTokenCount,
          total_tokens: response.usageMetadata?.totalTokenCount
        },
        finish_reason: this.mapFinishReason(response.candidates?.[0]?.finishReason),
        metadata: {
          provider: this.name,
          model: modelName,
          response_time_ms: responseTime
        }
      };
  }
  
  private convertMessagesToGemini(messages: LLMMessage[]): any[] {
    const convertedMessages: any[] = [];
    
    for (const message of messages) {
      if (message.role === 'system') {
        // Gemini doesn't have system role, convert to user message
        convertedMessages.push({
          role: 'user',
          parts: [{ text: `System: ${message.content}` }]
        });
      } else {
        convertedMessages.push({
          role: message.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: message.content }]
        });
      }
    }
    
    return convertedMessages;
  }
  
  private mapFinishReason(reason: any): 'stop' | 'tool_calls' | 'length' | 'content_filter' {
    switch (reason) {
      case 'STOP': return 'stop';
      case 'MAX_TOKENS': return 'length';
      case 'SAFETY': return 'content_filter';
      case 'RECITATION': return 'content_filter';
      default: return 'stop';
    }
  }
  
  getPricing(): { input_cost_per_token: number; output_cost_per_token: number; currency: string } {
    // Gemini pricing as of mid 2026 (in USD per million tokens, converted to per token)
    const pricingMap: Record<string, any> = {
      'gemini-3.7-flash': { input: 0.75, output: 3.75 },
      'gemini-3.6-flash': { input: 0.75, output: 3.75 },
      'gemini-3.5-flash-lite': { input: 0.30, output: 2.50 },
      'gemini-3.1-pro-preview': { input: 2.00, output: 12.00 }
    };

    const pricing = pricingMap[this.config.model] || pricingMap['gemini-3.7-flash'];
    
    return {
      input_cost_per_token: pricing.input / 1000000,
      output_cost_per_token: pricing.output / 1000000,
      currency: 'USD'
    };
  }
}