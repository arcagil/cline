import OpenAI from "openai"
import {
	cerebrasDefaultModelId,
	CerebrasModelId,
	cerebrasModels,
	ApiHandlerOptions,
	ModelInfo,
} from "../../shared/api"
import { ApiHandler } from "../index"
import { ApiStream } from "../transform/stream"
import { convertToOpenAiMessages } from "../transform/openai-format"

interface CompletionOptions {
	max_tokens?: number
	temperature?: number
	top_p?: number
	stream?: boolean
	echo?: boolean
	stop?: string[]
	user?: string
	seed?: number
}

export class CerebrasHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new OpenAI({
			apiKey: this.options.cerebrasApiKey,
			baseURL: "https://api.cerebras.ai/v1",
		})
	}

	async *createMessage(systemPrompt: string, messages: any[], options?: { jsonMode?: boolean }): ApiStream {
		const openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]

		const stream = await this.client.chat.completions.create({
			model: this.getModel().id,
			messages: openAiMessages,
			max_tokens: this.getModel().info.maxTokens,
			temperature: 0,
			stream: true,
			max_completion_tokens: this.getModel().info.maxTokens,
			seed: null,
			top_p: null,
			stop: null,
			tool_choice: "none", // Default to no tool usage
			tools: null,
			user: null,
			response_format: options?.jsonMode ? { type: "json_object" } : null,
			stream: options?.jsonMode ? false : true, // JSON mode is not compatible with streaming
		})

		if (options?.jsonMode) {
			// For JSON mode, we get a complete response
			const response = await stream as OpenAI.Chat.ChatCompletion
			if (response.choices[0]?.message?.content) {
				yield { type: "text", text: response.choices[0].message.content }
			}
		} else {
			// For streaming mode
			for await (const part of stream as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>) {
				const content = part.choices[0]?.delta?.content
				if (content) {
					yield { type: "text", text: content }
				}
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		const modelId = (this.options.cerebrasModelId as CerebrasModelId) || cerebrasDefaultModelId
		return {
			id: modelId,
			info: cerebrasModels[modelId],
		}
	}

	async *createCompletion(prompt: string | string[], options?: CompletionOptions): ApiStream {
		const stream = await this.client.completions.create({
			model: this.getModel().id,
			prompt,
			max_tokens: options?.max_tokens ?? this.getModel().info.maxTokens,
			temperature: options?.temperature ?? 0,
			top_p: options?.top_p ?? null,
			stream: options?.stream ?? true,
			echo: options?.echo ?? false,
			stop: options?.stop ?? null,
			user: options?.user ?? null,
			seed: options?.seed ?? null,
		})

		if (!options?.stream) {
			// For non-streaming mode
			const response = await stream as OpenAI.Completion
			if (response.choices[0]?.text) {
				yield { type: "text", text: response.choices[0].text }
			}
		} else {
			// For streaming mode
			for await (const part of stream as AsyncIterable<OpenAI.Completion>) {
				const text = part.choices[0]?.text
				if (text) {
					yield { type: "text", text }
				}
			}
		}
	}

	async listModels(): Promise<{ id: string; object: string; created: number; owned_by: string }[]> {
		const response = await this.client.models.list()
		return response.data.filter(model => 
			model.id === "llama3.1-8b" || model.id === "llama3.1-70b"
		)
	}

	async getModelInfo(modelId: string): Promise<{ 
		id: string; 
		object: string; 
		created: number; 
		owned_by: string 
	} | null> {
		try {
			const model = await this.client.models.retrieve(modelId)
			return model
		} catch (error) {
			console.error(`Error retrieving model ${modelId}:`, error)
			return null
		}
	}
}
