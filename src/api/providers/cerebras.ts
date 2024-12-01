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

	async *createMessage(systemPrompt: string, messages: any[]): ApiStream {
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
		})

		for await (const part of stream) {
			const content = part.choices[0]?.delta?.content
			if (content) {
				yield { type: "text", text: content }
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
}
