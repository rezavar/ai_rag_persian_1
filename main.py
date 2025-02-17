MODEL = 'qwen2.5:latest'

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings


# بارگذاری اسناد
directory = 'dataset/pdfFiles'
loader = PyPDFDirectoryLoader(directory)
data = loader.load_and_split()

# مدل‌های مورد نیاز
embedding = OllamaEmbeddings(model=MODEL)
output_parser = StrOutputParser()
model = ChatOllama(model=MODEL)

# تعریف قالب پرسش
template = """
Answer the question based on the context below. 
لطفاً فقط به زبان فارسی پاسخ دهید.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# ذخیره اسناد در وکتور استور
vector_store = DocArrayInMemorySearch.from_documents(documents=data, embedding=embedding)
retriever = vector_store.as_retriever()

question = "بعد از دزدیدن گوشی چیکار باید بکنم"
# دریافت اسناد مرتبط
retrieved_docs = retriever.get_relevant_documents(question)
context_text = "\n".join([doc.page_content for doc in retrieved_docs])

# اجرای زنجیره پردازش
chain = prompt | model | output_parser
x = chain.invoke({'context': context_text, 'question': question})

print(x)
