import os
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceTextGenInference
from langchain_community.chat_models.huggingface import ChatHuggingFace

output_parser = StrOutputParser()


class LLM(ABC):
    def __init__(self, name: str):
        self.name = name
        self.llm = None
        self.embeddings = None

    def get_name(self):
        return self.name

    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings


class OpenAIModel(LLM):
    def __init__(self, name: str, api_key: str = None):
        super().__init__(name)
        self.llm = ChatOpenAI(model=self.name, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings()


class OllamaModel(LLM):
    def __init__(self, name: str):
        super().__init__(name)
        self.llm = Ollama(model=self.name)
        self.embeddings = OllamaEmbeddings()


class HFModel(LLM):
    def __init__(self, name, token):
        super().__init__(name)
        self.llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
            huggingfacehub_api_token=f"{token}",
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            top_k=30,
            temperature=0.1,
            repetition_penalty=1.03,

        ))
        self.embeddings = HuggingFaceEmbeddings()


class Docs:
    def __init__(self):
        self.docs = []

    def set_pdf_docs(self, filepath):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        self.docs = docs

    def get_docs(self):
        return self.docs


class VectorStore:
    def __init__(self, docs: Docs):
        self.docs = docs
        self.vector = None

    def set_vector(self, embeddings):
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(self.docs.get_docs())
        vector = FAISS.from_documents(documents, embeddings)
        self.vector = vector

    def get_vector(self):
        return self.vector


class Chain:
    def __init__(self, model: LLM):
        self.model = model
        self.chain = None

    def set_simple_chain(self):
        self.chain = self.model.llm | output_parser

    def set_conversational_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, answer",
                ),
            ]
        )

        self.chain = prompt | self.model.llm | output_parser

    def set_retrieval_chain(self, vector: VectorStore):
        retriever = vector.get_vector().as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )
        retriever_chain = create_history_aware_retriever(self.model.llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(self.model.llm, prompt)

        self.chain = create_retrieval_chain(retriever_chain, document_chain)

    def get_chain(self):
        return self.chain


class ChatBot:
    def __init__(self, chain: Chain):
        self.chat_history = []
        self.chain = chain

    def get_simple_answer(self, message):
        answer = self.chain.get_chain().invoke(message)
        self.add_history(message, answer)
        return answer

    def get_conversational_answer(self, message):
        answer = self.chain.get_chain().invoke(
            {"chat_history": self.chat_history, "input": message}
        )
        self.add_history(message, answer)
        return answer

    def get_retrieval_answer(self, message):
        answer = self.chain.get_chain().invoke(
            {"chat_history": self.chat_history, "input": message}
        )["answer"]
        self.add_history(message, answer)
        return answer

    def add_history(self, message, answer):
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=answer))


class ChatbotController:
    def __init__(self, model: LLM):
        self.model = model
        self.model_types = {
            "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
            "Ollama": ["llama2:latest", "mistral:latest", "gemma:latest"],
            "HuggingFace": ["HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-v0.1"]
        }

    def set_model(self, model_type, model_ref, api_token=None):
        if model_type == "OpenAI":
            self.model = OpenAIModel(model_ref, api_token)
        elif model_type == "HuggingFace":
            self.model = HFModel(model_ref, api_token)
        else:
            self.model = OllamaModel(model_ref)

        return "Done"


class ChatbotView:
    def __init__(self, controller: ChatbotController):
        self.controller = controller

        with gr.Blocks() as demo:
            model_type = gr.Radio(value="OpenAI", label="Model Source", choices=["OpenAI", "Ollama", "HuggingFace"])
            api_token = gr.Text("OpenAI")
            selected_model = gr.Dropdown(value="gpt-3.5-turbo", choices=self.controller.model_types["OpenAI"])
            progress_bar = gr.Label("Upload your PDF")

            model_type.change(self.filter_model_types, model_type, [api_token, selected_model])
            selected_model.change(self.controller.set_model, inputs=[model_type, selected_model], outputs=progress_bar)

        self.gui = demo

    def filter_model_types(self, model_type):
        return gr.Text(model_type), gr.Dropdown(value=self.controller.model_types[model_type][0],
                                                choices=self.controller.model_types[model_type])

    def get_gui(self):
        return self.gui



if __name__ == "__main__":
    chatbot_controller = ChatbotController(OpenAIModel("gpt-3.5-turbo"))
    chatbot_view = ChatbotView(chatbot_controller)
    chatbot_view.get_gui().launch()


    # model_name = input("Ingrese el nombre del modelo (por ejemplo, OpenAI o OLLAMA): ")
    # chatbot_model = OllamaModel("llama2")
    #
    # if model_name.lower() == "openai":
    #     openai_api_key = input("Input ypur OpenAI Api Key, see on https://platform.openai.com/api-keys: ")
    #     model_ref = input("input your model ref, see on https://platform.openai.com/docs/models/gpt-3-5-turbo: ")
    #     chatbot_model = OpenAIModel(model_ref, openai_api_key)
    # elif model_name.lower() == "hf":
    #     hf_token = input("input your HF token see on https://huggingface.co/settings/tokens:")
    #     model_ref = input("input your model ref: see on https://huggingface.co/models?pipeline_tag=text-generation&sort=trending")
    #     chatbot_model = HFModel(model_ref, hf_token)
    # elif model_name.lower() == "ollama":
    #     model_ref = input("input your model ref: ")
    #     chatbot_model = OllamaModel(model_ref)
    # else:
    #     print("Nombre de modelo no válido.")
    #     exit()
    #
    # user_docs = Docs()
    # user_docs.set_pdf_docs(
    #     "/home/jorge/Development/chat_with_PDF/La temporada invernal será más húmeda de lo normal y con precipitaciones.pdf")
    #
    # vectorstore = VectorStore(user_docs)
    # vectorstore.set_vector(chatbot_model.get_embeddings())
    #
    # chatbot_chain = Chain(chatbot_model)
    # chatbot_chain.set_retrieval_chain(vectorstore)
    #
    # chatbot = ChatBot(chatbot_chain)
    #
    # while True:
    #     user_message = input("Send your message:")
    #     ai_answer = chatbot.get_retrieval_answer(user_message)
    #     print(ai_answer)
