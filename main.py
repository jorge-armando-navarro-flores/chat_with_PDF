import os
import gradio as gr
from langchain_openai import ChatOpenAI
import subprocess
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
import faiss


class Chatbot:
    def __init__(self):
        self.openai = ""
        print(not self.openai)
        self.chat_history = []
        self.docs = []
        self.embeddings = OllamaEmbeddings()
        self.vector = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatIP(768),
            docstore=None,
            index_to_docstore_id={}
        )
        self.model = "llama2:latest"
        self.llm = Ollama(model=self.model)
        self.chain = Ollama(model=self.model)
        self.retrieval_chain = False

    def set_openai(self, openai_api):
        os.environ["OPENAI_API_KEY"] = openai_api
        self.openai = openai_api
        self.embeddings = OpenAIEmbeddings()
        self.model = "gpt-3.5-turbo"
        self.llm = ChatOpenAI(model=self.model)
        self.chain = ChatOpenAI(model=self.model) | output_parser

    def set_opensource(self):
        self.embeddings = OllamaEmbeddings()
        self.model = "llama2:latest"
        self.llm = Ollama(model=self.model)
        self.chain = Ollama(model=self.model)
        self.retrieval_chain = False

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        return "Upload and Process Again"

    def get_docs(self, filepath):
        loader = PyMuPDFLoader(filepath)
        docs = loader.load()
        self.docs = docs

    def get_vector(self):

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(self.docs)
        self.vector = FAISS.from_documents(documents, self.embeddings)

    def get_chain(self):
        retriever = self.vector.as_retriever()
        self.llm = Ollama(model=self.model) if not self.openai else ChatOpenAI(model=self.model)

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
        retriever_chain = create_history_aware_retriever(self.llm, retriever, prompt)

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
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        self.chain = create_retrieval_chain(retriever_chain, document_chain)
        self.retrieval_chain = True

    def undo(self):
        self.chat_history.pop()
        self.chat_history.pop()

    def clear(self):
        self.chat_history = []

    def predict(self, message, history):

        if self.retrieval_chain:
            response = self.chain.invoke(
                {"chat_history": self.chat_history, "input": message}
            )["answer"]
        else:
            response = self.chain.invoke(
                message
            )

        # self.get_apa_reference(response)

        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=response))

        return response

    def process(self, filepath):
        self.get_docs(filepath)
        self.get_vector()
        self.get_chain()
        return "Done"

    def get_apa_reference(self, response):
        r_docs = self.vector.similarity_search(response, k=2)
        print(r_docs)

    def get_models(self):
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        open_models = [m.split()[0] for m in result.stdout.decode('utf-8').split("\n")[1:-1]]
        openai_models = ["gpt-3.5-turbo", "gpt-4"]
        return open_models if not self.openai else openai_models


chatbot = Chatbot()

undo_button = gr.Button("↩️ Undo")
clear_button = gr.Button("🗑️  Clear")

with gr.Blocks() as demo:
    undo_button.click(chatbot.undo)
    clear_button.click(chatbot.clear())
    gr.ChatInterface(chatbot.predict, retry_btn="🔄  Retry", undo_btn=undo_button, clear_btn=clear_button)

    progress_bar = gr.Label("Upload your PDF")
    file = gr.File(file_types=[".pdf"])

    models = gr.Radio(label="Model Source", choices=["OpenAI", "Open Source"])
    openai_api_text = gr.Text(placeholder="Open AI API key", visible=False, interactive=True, type="password")
    selected_model = gr.Dropdown(label="Model", choices=[])

    models_map = {
        "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
        "Open Source": chatbot.get_models(),
    }


    def filter_models(species):
        visible = species == "OpenAI"
        if not visible:
            chatbot.set_opensource()
        return gr.Dropdown(
            choices=models_map[species], value=models_map[species][0]
        ), gr.Text(visible=visible)

    openai_api_text.input(chatbot.set_openai, openai_api_text)
    models.change(filter_models, models, [selected_model, openai_api_text])

    selected_model.change(chatbot.set_model, inputs=selected_model, outputs=progress_bar)
    process_btn = gr.Button("Process")
    process_btn.click(chatbot.process, inputs=file, outputs=progress_bar)

if __name__ == "__main__":
    demo.queue().launch()
