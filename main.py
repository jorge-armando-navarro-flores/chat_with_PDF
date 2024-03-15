import os
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
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
from langchain_community.llms import HuggingFaceEndpoint

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
    def __init__(self, name: str, api_key: str):
        super().__init__(name)
        self.llm = ChatOpenAI(model=self.name, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings()


class OllamaModel(LLM):
    def __init__(self, name: str):
        super().__init__(name)
        self.llm = Ollama(model=self.name)
        self.embeddings = OllamaEmbeddings()


class HFModel(LLM):
    def __init__(self, name, endpoint_url, token):
        super().__init__(name)
        self.llm = HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            max_new_tokens=1,
            top_k=50,
            temperature=0.1,
            repetition_penalty=1.03,
            huggingfacehub_api_token=token
        )
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=token, model_name="sentence-transformers/all-MiniLM-l6-v2"
        )


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


if __name__ == "__main__":

    model_name = input("Ingrese el nombre del modelo (por ejemplo, OpenAI o OLLAMA): ")
    chatbot_model = OllamaModel("llama2")

    if model_name.lower() == "openai":
        openai_api_key = input("Ingrese la API key de OpenAI: ")
        chatbot_model = OpenAIModel("gpt-3.5-turbo", openai_api_key)
    elif model_name.lower() == "hf":
        hf_endpoint_url = input("input your HF endpoint url: ")
        hf_token = input("input your HF token: ")
        chatbot_model = HFModel("hf-model", hf_endpoint_url, hf_token)
    elif model_name.lower() == "ollama":
        chatbot_model = OllamaModel("llama2")
    else:
        print("Nombre de modelo no válido.")
        exit()

    user_docs = Docs()
    user_docs.set_pdf_docs(
        "/home/jorge/Development/chat_with_PDF/La temporada invernal será más húmeda de lo normal y con precipitaciones.pdf")

    vectorstore = VectorStore(user_docs)
    vectorstore.set_vector(chatbot_model.get_embeddings())

    chatbot_chain = Chain(chatbot_model)
    chatbot_chain.set_retrieval_chain(vectorstore)

    chatbot = ChatBot(chatbot_chain)

    while True:
        user_message = input("Send your message:")
        ai_answer = chatbot.get_retrieval_answer(user_message)
        print(ai_answer)
