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

OPENAI_API_KEY = None # os.environ.get("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
subprocess.run(['ollama', 'pull', 'gemma'])


class Chatbot:
    def __init__(self):
        self.not_openai = not OPENAI_API_KEY
        self.chat_history = []
        self.docs = []
        self.embeddings = OllamaEmbeddings() if self.not_openai else OpenAIEmbeddings()
        self.vector = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatIP(768),
            docstore=None,
            index_to_docstore_id={}
        )
        self.model = "llama2:latest" if self.not_openai else "gpt-3.5-turbo"
        self.llm = Ollama(model=self.model) if self.not_openai else ChatOpenAI(model=self.model)
        self.chain = Ollama(model=self.model) if self.not_openai else ChatOpenAI(model=self.model) | output_parser
        self.retrieval_chain = False

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        return "Process Again"

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
        self.llm = Ollama(model=self.model) if self.not_openai else ChatOpenAI(model=self.model)

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
        return open_models if self.not_openai else openai_models


chatbot = Chatbot()

undo_button = gr.Button("↩️ Undo")
clear_button = gr.Button("🗑️  Clear")

with gr.Blocks() as demo:
    undo_button.click(chatbot.undo)
    clear_button.click(chatbot.clear())
    gr.ChatInterface(chatbot.predict, retry_btn="🔄  Retry", undo_btn=undo_button, clear_btn=clear_button)

    progress_bar = gr.Label("Upload your PDF")
    file = gr.File(file_types=[".pdf"])
    selected_model = gr.Dropdown(value=chatbot.model, choices=chatbot.get_models(), container=False)
    selected_model.change(chatbot.set_model, inputs=selected_model, outputs=progress_bar)
    process_btn = gr.Button("Process")
    process_btn.click(chatbot.process, inputs=file, outputs=progress_bar)

if __name__ == "__main__":
    demo.queue().launch()
