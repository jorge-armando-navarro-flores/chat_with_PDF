import os
import gradio as gr
from langchain_openai import ChatOpenAI
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


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
not_openai = not OPENAI_API_KEY


class Chatbot:
    def __init__(self):
        self.chat_history = []
        self.docs = []
        self.chain = ChatOpenAI()

    def get_docs(self, url):
        loader = PyMuPDFLoader(url)
        docs = loader.load()
        self.docs = docs

    def get_vector(self):
        embeddings = OllamaEmbeddings() if not_openai else OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(self.docs)
        vector = FAISS.from_documents(documents, embeddings)
        return vector

    def get_chain(self):
        vector = self.get_vector()
        retriever = vector.as_retriever()
        llm = Ollama(model="llama2") if not_openai else ChatOpenAI()

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
        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

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
        document_chain = create_stuff_documents_chain(llm, prompt)

        self.chain = create_retrieval_chain(retriever_chain, document_chain)

    def predict(self, message):

        response = self.chain.invoke(
            {"chat_history": self.chat_history, "input": message}
        )
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=response["answer"]))

        return response["answer"]


chatbot = Chatbot()
chatbot.get_docs("/home/jorge/Development/chat_with_PDF/La temporada invernal será más húmeda de lo normal y con precipitaciones.pdf")
chatbot.get_chain()

print(chatbot.predict("Who is Mauricio?"))
print()
print(chatbot.predict("What did he say?"))
